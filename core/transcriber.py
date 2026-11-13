import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

import whisper

from utils.audio_processor import _run_ffmpeg, _send_to_sarvam

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25-second pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


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


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    base_dir = os.path.dirname(chunk_path) or "."
    base_name = os.path.splitext(os.path.basename(chunk_path))[0]
    segment_pattern = os.path.join(base_dir, f"{base_name}_sv_%03d.wav")

    _run_ffmpeg([
        "-y",
        "-i",
        chunk_path,
        "-f",
        "segment",
        "-segment_time",
        str(SARVAM_PIECE_SECONDS),
        "-c:a",
        "pcm_s16le",
        segment_pattern,
    ])

    piece_paths = sorted(os.path.join(base_dir, f"{base_name}_sv_{i:03d}.wav") for i in range(0, 9999))
    valid_piece_paths = [path for path in piece_paths if os.path.exists(path)]

    full_text = ""
    total_pieces = len(valid_piece_paths)
    for i, piece in enumerate(valid_piece_paths, start=1):
        try:
            print(f"  → Sarvam piece {i}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece) + " "
        finally:
            if os.path.exists(piece):
                os.remove(piece)

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
