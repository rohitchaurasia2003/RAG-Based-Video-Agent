import unittest


class ProjectSmokeTests(unittest.TestCase):
    def test_project_modules_import(self):
        import app
        import main
        import utils.audio_processor
        import core.transcriber
        import core.summarizer
        import core.extractor
        import core.rag_engine
        import core.vector_store


if __name__ == "__main__":
    unittest.main()
