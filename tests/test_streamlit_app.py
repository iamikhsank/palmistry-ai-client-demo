"""Real Streamlit smoke tests; no photos or model inference are required."""
import importlib.util
import unittest
from pathlib import Path

HAS_STREAMLIT = importlib.util.find_spec('streamlit') is not None


@unittest.skipUnless(HAS_STREAMLIT, 'Streamlit is not installed')
class StreamlitAppTests(unittest.TestCase):
    def test_two_hand_upload_and_camera_pages_render(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        app = AppTest.from_file(str(app_path), default_timeout=40).run()
        self.assertFalse(app.exception, list(app.exception))
        self.assertTrue(any('Analisis dua telapak tangan' in x.value for x in app.subheader))
        source = next(x for x in app.radio if x.label == 'Sumber foto')
        self.assertEqual(source.value, 'Unggah gambar')
        self.assertTrue(next(x for x in app.button if x.label == 'Analisis foto').disabled)

        source.set_value('Kamera').run()
        self.assertFalse(app.exception, list(app.exception))
        self.assertEqual(next(x for x in app.radio if x.label == 'Sumber foto').value, 'Kamera')
        self.assertTrue(next(x for x in app.button if x.label == 'Analisis foto').disabled)

    def test_existing_prediction_page_remains_accessible(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        app = AppTest.from_file(str(app_path), default_timeout=40).run()
        next(x for x in app.radio if x.label == 'Navigasi').set_value('Live Prediction').run()
        self.assertFalse(app.exception, list(app.exception))
        self.assertTrue(any('Coba model' in x.value for x in app.subheader))


if __name__ == '__main__':
    unittest.main()
