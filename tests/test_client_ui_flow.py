"""Exercise session-state behaviour without Streamlit or model dependencies."""
import hashlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Box:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Photo:
    def __init__(self, data):
        self.data = data

    def getvalue(self):
        return self.data


class FakeStreamlit(types.ModuleType):
    def __init__(self, photos, session, mode='Unggah gambar'):
        super().__init__('streamlit')
        self.photos = photos
        self.session_state = session
        self.mode = mode
        self.exported = None

    def subheader(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def caption(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def image(self, *args, **kwargs): pass
    def dataframe(self, *args, **kwargs): pass
    def radio(self, *args, **kwargs): return self.mode
    def slider(self, *args, **kwargs): return 0.5
    def button(self, *args, **kwargs): return False
    def columns(self, count): return [Box() for _ in range(count)]
    def expander(self, *args, **kwargs): return Box()

    def file_uploader(self, label, **kwargs):
        return self.photos.get(kwargs['key'].removeprefix('upload_'))

    def camera_input(self, label, **kwargs):
        return self.photos.get(kwargs['key'].removeprefix('camera_'))

    def selectbox(self, label, choices, *, key, **kwargs):
        return self.session_state.setdefault(key, choices[0])

    def download_button(self, label, data, **kwargs):
        self.exported = data


def load_ui(st):
    runtime = types.ModuleType('model_runtime')
    runtime.PalmImageProcessor = object
    runtime.predict_image = lambda *args: None
    crop = types.ModuleType('modules.crop_preview')
    crop.center_crop_preview = lambda data: (data, 0.0)
    pipeline = types.ModuleType('modules.palm_dl_pipeline')
    pipeline.TARGET_PALM_FEATURES = []
    modules = {'streamlit': st, 'model_runtime': runtime,
               'modules.crop_preview': crop, 'modules.palm_dl_pipeline': pipeline}
    path = Path(__file__).resolve().parents[1] / 'modules' / 'client_ui.py'
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('client_ui_test', path)
        ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui)
    return ui


def signature(photos):
    return (0.5, tuple((side, hashlib.sha256(photo.getvalue()).hexdigest())
                       for side, photo in sorted(photos.items())))


def result(narrative):
    return {'narrative': narrative, 'quality': {'sharpness': 100, 'brightness': 120},
            'ridge': b'', 'scores': {}}


class ClientUIFlowTests(unittest.TestCase):
    def test_two_hands_export_keeps_manual_and_model_text(self):
        photos = {'kiri': Photo(b'left'), 'kanan': Photo(b'right')}
        state = {'client_signature': signature(photos),
                 'client_results': {'kiri': result('MODEL KIRI'), 'kanan': result('MODEL KANAN')},
                 'manual_kiri_kehidupan': 'samar', 'manual_kanan_kepala': 'jelas'}
        st = FakeStreamlit(photos, state)
        load_ui(st).render_client_analysis(lambda: None)
        self.assertIn('MODEL KIRI', st.exported)
        self.assertIn('MODEL KANAN', st.exported)
        self.assertIn('Garis kehidupan: terlihat samar', st.exported)
        self.assertIn('Garis kepala: terlihat jelas', st.exported)
        self.assertIn('diisi peninjau secara manual', st.exported)

    def test_changed_photo_clears_prior_manual_and_model_results(self):
        old = {'kiri': Photo(b'old')}
        new = {'kiri': Photo(b'new')}
        state = {'client_signature': signature(old), 'client_results': {'kiri': result('OLD')},
                 'manual_kiri_kehidupan': 'samar', 'manual_kanan_kepala': 'jelas'}
        st = FakeStreamlit(new, state)
        load_ui(st).render_client_analysis(lambda: None)
        self.assertNotIn('client_results', state)
        self.assertNotIn('manual_kiri_kehidupan', state)
        self.assertNotIn('manual_kanan_kepala', state)
        self.assertIsNone(st.exported)

    def test_camera_photos_keep_same_session_results(self):
        photos = {'kiri': Photo(b'camera-left'), 'kanan': Photo(b'camera-right')}
        state = {'client_signature': signature(photos),
                 'client_results': {'kiri': result('MODEL KIRI'), 'kanan': result('MODEL KANAN')},
                 'manual_kiri_asmara': 'samar'}
        st = FakeStreamlit(photos, state, mode='Kamera')
        load_ui(st).render_client_analysis(lambda: None)
        self.assertIn('Garis asmara: terlihat samar', st.exported)
        self.assertEqual(state['manual_kiri_asmara'], 'samar')


if __name__ == '__main__':
    unittest.main()
