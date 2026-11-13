# AI Video Assistant

Turn a meeting recording or YouTube video into a transcript, concise summary, action items, decisions, open questions, and a chat interface grounded in the transcript.

## Features

- Process a YouTube URL, a local audio/video path, or an uploaded media file.
- Transcribe English audio locally with OpenAI Whisper.
- Transcribe Hinglish audio with Sarvam Speech-to-Text Translate, returning English text.
- Generate a meeting title, summary, action items, key decisions, and open questions with Groq.
- Ask follow-up questions using retrieval-augmented generation (RAG) over the transcript.
- Use the Streamlit interface or the command-line pipeline.

## Requirements

- Python 3.10 or newer
- FFmpeg available on your `PATH`
- A Groq API key
- A Sarvam API key for Hinglish transcription

Install FFmpeg if it is not already installed:

```bash
# macOS with Homebrew
brew install ffmpeg

# Debian or Ubuntu
sudo apt update && sudo apt install ffmpeg
```

On Windows, install an FFmpeg build and add its `bin` directory to `PATH`.

## Setup

From the project directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate          # Windows PowerShell
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your own credentials:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```

`SARVAM_API_KEY` is only needed when you choose Hinglish. Never commit `.env` or paste API keys into source code, issue reports, or chat messages. The project `.gitignore` excludes `.env`.

## Run the app

Start the Streamlit interface from the project directory:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit. In the sidebar, enter a YouTube URL or local media path, or upload a media file. Use only one input at a time, select `english` or `hinglish`, and choose **Analyse**. Uploading is useful when a remote host cannot download a video from YouTube. The first Whisper run may download the selected model weights.

When analysis completes, the app displays the title, summary, full transcript, action items, key decisions, and open questions. Use the chat area to ask questions about the transcript.

### Command line

The same analysis pipeline is available without Streamlit:

```bash
python main.py
```

Enter a YouTube URL or local file path when prompted, then choose `english` or `hinglish`. The CLI prints the analysis and opens a transcript Q&A loop; enter `exit` to leave it.

## YouTube downloads

YouTube may reject downloads with HTTP 403, even when `yt-dlp` is up to date. If you are authorized to view the video in a local browser, you can optionally let `yt-dlp` use that browser's cookies. Add one of the following to `.env`, matching a browser profile available on this computer:

```env
YOUTUBE_COOKIES_FROM_BROWSER=chrome
```

Supported browser names include `chrome`, `firefox`, `edge`, and `brave`. The browser profile must be available on the machine running Streamlit; a hosted deployment cannot access browser cookies from your computer. When running remotely, upload the media file instead, or set `YOUTUBE_COOKIES_FILE` to the path of a Netscape-format cookies file available to the server:

```env
YOUTUBE_COOKIES_FILE=/absolute/path/to/cookies.txt
```

Restart Streamlit after changing `.env`. Cookies grant access to your account: keep them private and do not commit or share cookie files. Cookie authentication cannot bypass private-video permissions, regional restrictions, or other access controls.

## Data and privacy

- English audio is transcribed locally by Whisper.
- Hinglish audio is sent to Sarvam for speech-to-text translation.
- Transcript text is sent to Groq for title generation, summarization, extraction, and question answering.
- Downloaded YouTube media is stored under `downloads/`.
- Transcript embeddings are stored locally by Chroma under `vector_db/`.

Use recordings and credentials only when you have permission to process them. Remove local data directories when you no longer need the downloaded media or vector data.

## Project layout

```text
app.py                    Streamlit UI and analysis workflow
main.py                   Command-line workflow
core/transcriber.py       Whisper and Sarvam transcription
core/summarizer.py        Groq title generation and summaries
core/extractor.py         Action items, decisions, and questions
core/rag_engine.py        Transcript question answering
core/vector_store.py      Chroma vector database
utils/audio_processor.py  Download, conversion, and audio chunking
tests/                    Regression and smoke tests
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
