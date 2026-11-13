import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

import requests
import whisper
import yt_dlp

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25-second pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

_model = None


def _resolve_downloaded_media_path(directory: str) -> str:
    directory_path = Path(directory)
    media_extensions = (".wav", ".webm", ".m4a", ".mp4", ".mp3", ".aac")
    candidates = []

    for ext in media_extensions:
        candidates.extend(directory_path.glob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(f"No downloaded media file found in '{directory}'.")

    wav_candidates = [path for path in candidates if path.suffix.lower() == ".wav"]
    if wav_candidates:
        candidates = wav_candidates

    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(newest)


def save_uploaded_media(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="upload_",
        suffix=suffix,
        dir=DOWNLOAD_DIR,
        delete=False,
    ) as media_file:
        media_file.write(uploaded_file.getbuffer())
    return media_file.name


def ensure_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is required for audio conversion and chunking but was not found on PATH. "
            "Install FFmpeg and try again."
        )
    return ffmpeg_path


def _run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    ffmpeg = ensure_ffmpeg()
    command = [ffmpeg, *args]
    completed = subprocess.run(command, capture_output=True, text=True)
    if check and completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"FFmpeg command failed: {' '.join(command)}\n{stderr or 'unknown ffmpeg error'}")
    return completed


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    base_dir = os.path.dirname(chunk_path) or "."
    base_name = Path(chunk_path).stem
    piece_duration_seconds = SARVAM_PIECE_SECONDS
    segment_pattern = os.path.join(base_dir, f"{base_name}_sv_%03d.wav")

    _run_ffmpeg([
        "-y",
        "-i",
        chunk_path,
        "-f",
        "segment",
        "-segment_time",
        str(piece_duration_seconds),
        "-c:a",
        "pcm_s16le",
        segment_pattern,
    ])

    piece_paths = sorted(Path(base_dir).glob(f"{base_name}_sv_*.wav"))
    total_pieces = len(piece_paths)
    full_text = ""

    for i, piece in enumerate(piece_paths, start=1):
        try:
            print(f"  → Sarvam piece {i}/{total_pieces} ...")
            full_text += _send_to_sarvam(str(piece)) + " "
        finally:
            if piece.exists():
                piece.unlink()

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    if not chunks:
        return ""

    full_transcript = ""
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    cookies_file = os.getenv("YOUTUBE_COOKIES_FILE")
    browser = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER")
    if cookies_file:
        ydl_opts["cookiefile"] = str(Path(cookies_file).expanduser())
    elif browser:
        ydl_opts["cookiesfrombrowser"] = (browser.strip().lower(),)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        if "403" in str(exc) or "forbidden" in str(exc).lower():
            raise RuntimeError(
                "YouTube denied the download (HTTP 403). If you can view this video in your browser, "
                "set YOUTUBE_COOKIES_FROM_BROWSER=chrome (or firefox, edge, or brave) in .env and "
                "restart the app. Alternatively, set YOUTUBE_COOKIES_FILE to a Netscape-format "
                "cookies file. On hosted deployments, the browser profile on your computer is not "
                "available to the app; upload the media file in the sidebar instead."
            ) from exc
        raise

    downloaded_path = _resolve_downloaded_media_path(DOWNLOAD_DIR)
    if not downloaded_path:
        raise RuntimeError(f"Download failed for URL: {url}")

    return downloaded_path


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using FFmpeg."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    _run_ffmpeg([
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ])
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    base_dir = os.path.dirname(wav_path) or "."
    base_name = Path(wav_path).stem
    pattern = os.path.join(base_dir, f"{base_name}_chunk_%03d.wav")

    _run_ffmpeg([
        "-y",
        "-i",
        wav_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_minutes * 60),
        "-c:a",
        "pcm_s16le",
        pattern,
    ])

    chunk_files = sorted(Path(base_dir).glob(f"{base_name}_chunk_*.wav"))
    if not chunk_files:
        raise RuntimeError(f"Failed to create audio chunks for {wav_path}")
    return [str(path) for path in chunk_files]


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks