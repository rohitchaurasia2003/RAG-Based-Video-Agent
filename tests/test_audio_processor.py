import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils import audio_processor


class AudioProcessorTests(unittest.TestCase):
    def test_resolve_downloaded_media_path_prefers_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "PM Modi's most expensive policy failure | Let Me Explain.webm")
            final = os.path.join(tmpdir, "PM Modi's most expensive policy failure | Let Me Explain.wav")

            with open(original, "wb") as f:
                f.write(b"raw")
            with open(final, "wb") as f:
                f.write(b"converted")

            resolved = audio_processor._resolve_downloaded_media_path(tmpdir)
            self.assertEqual(resolved, final)

    def test_save_uploaded_media_writes_to_download_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            uploaded_file = Mock()
            uploaded_file.name = "meeting.mp4"
            uploaded_file.getbuffer.return_value = b"media bytes"

            with patch.object(audio_processor, "DOWNLOAD_DIR", tmpdir):
                saved_path = audio_processor.save_uploaded_media(uploaded_file)

            self.assertEqual(os.path.dirname(saved_path), tmpdir)
            self.assertTrue(saved_path.endswith(".mp4"))
            with open(saved_path, "rb") as media_file:
                self.assertEqual(media_file.read(), b"media bytes")

    @patch.dict(os.environ, {"YOUTUBE_COOKIES_FROM_BROWSER": "chrome", "YOUTUBE_COOKIES_FILE": ""})
    @patch.object(audio_processor, "_resolve_downloaded_media_path", return_value="downloads/video.wav")
    @patch.object(audio_processor.yt_dlp, "YoutubeDL")
    def test_download_youtube_audio_uses_configured_browser_cookies(
        self, youtube_dl, resolve_downloaded_path
    ):
        audio_processor.download_youtube_audio("https://www.youtube.com/watch?v=example")

        options = youtube_dl.call_args.args[0]
        self.assertEqual(options["cookiesfrombrowser"], ("chrome",))
        youtube_dl.return_value.__enter__.return_value.extract_info.assert_called_once()

    @patch.object(audio_processor.yt_dlp, "YoutubeDL")
    def test_download_youtube_audio_explains_http_403(self, youtube_dl):
        youtube_dl.return_value.__enter__.return_value.extract_info.side_effect = (
            audio_processor.yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")
        )

        with self.assertRaisesRegex(RuntimeError, "YOUTUBE_COOKIES_FROM_BROWSER"):
            audio_processor.download_youtube_audio("https://www.youtube.com/watch?v=example")


if __name__ == "__main__":
    unittest.main()
