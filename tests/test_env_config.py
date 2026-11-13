import os
import unittest
import importlib

from core import summarizer


class EnvConfigTests(unittest.TestCase):
    def test_groq_key_accepts_typoed_env_name(self):
        os.environ["GORQ_API_KEY"] = "test-key"
        os.environ.pop("GROQ_API_KEY", None)
        llm = summarizer.get_llm()
        self.assertIsNotNone(llm)

    def test_sarvam_key_from_dotenv_overrides_stale_shell_value(self):
        os.environ["SARVAM_API_KEY"] = "stale-shell-key"

        import utils.audio_processor as audio_processor
        importlib.reload(audio_processor)

        self.assertTrue(audio_processor.SARVAM_API_KEY)
        self.assertNotEqual(audio_processor.SARVAM_API_KEY, "stale-shell-key")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
