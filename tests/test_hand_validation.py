import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageFilter

from modules.hand_validation import (
    HandDetectorUnavailable, HandNotDetected, estimate_sharpness, validate_hand_image,
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

    def test_sharpness_drops_on_blurred_pattern(self):
        pattern = Image.new("L", (512, 512))
        for x in range(512):
            for y in range(512):
                pattern.putpixel((x, y), 255 if (x // 6) % 2 else 0)
        def encode(image):
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            return stream.getvalue()
        sharp = estimate_sharpness(encode(pattern))
        blurred = estimate_sharpness(encode(pattern.filter(ImageFilter.GaussianBlur(12))))
        self.assertGreater(sharp, blurred)
        self.assertLess(blurred, 2.0)

    def test_blank_is_low_sharpness(self):
        self.assertEqual(estimate_sharpness(image_bytes()), 0.0)

    def test_real_detector_rejects_blank_if_installed(self):
        try:
            import mediapipe
        except ImportError:
            self.skipTest("MediaPipe is not installed")
        with self.assertRaises(HandNotDetected):
            validate_hand_image(image_bytes((320, 320)))


if __name__ == "__main__":
    unittest.main()
