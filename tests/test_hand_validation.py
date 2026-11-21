import io
import unittest
from unittest.mock import patch

from PIL import Image

from modules.hand_validation import (
    HandDetectorUnavailable, HandNotDetected, validate_hand_image,
)


def image_bytes(size=(80, 40)):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format="PNG")
    return stream.getvalue()


class HandValidationTests(unittest.TestCase):
    def test_detector_receives_center_square(self):
        seen = []
        result = validate_hand_image(image_bytes(), lambda rgb: seen.append(rgb.shape) or True)
        self.assertTrue(result.detected)
        self.assertEqual(seen, [(40, 40, 3)])

    def test_detected_hand_is_accepted(self):
        self.assertTrue(validate_hand_image(image_bytes(), lambda rgb: True).detected)

    def test_image_without_hand_is_rejected(self):
        with self.assertRaisesRegex(HandNotDetected, "Tangan tidak terdeteksi"):
            validate_hand_image(image_bytes(), lambda rgb: False)

    def test_detector_failure_is_not_treated_as_no_hand(self):
        def broken(rgb):
            raise RuntimeError("unavailable")
        with self.assertRaises(HandDetectorUnavailable):
            validate_hand_image(image_bytes(), broken)

    def test_corrupt_image_does_not_reach_detector(self):
        seen = []
        with self.assertRaises(Exception):
            validate_hand_image(b"not-an-image", lambda rgb: seen.append(rgb))
        self.assertFalse(seen)

    def test_real_detector_rejects_blank_if_installed(self):
        try:
            import mediapipe
        except ImportError:
            self.skipTest("MediaPipe is not installed")
        with self.assertRaises(HandNotDetected):
            validate_hand_image(image_bytes((320, 320)))


if __name__ == "__main__":
    unittest.main()
