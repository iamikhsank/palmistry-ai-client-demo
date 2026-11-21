"""Reject images without a detectable hand before the palm model runs."""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


class HandNotDetected(ValueError):
    """No hand landmarks could be detected in the region sent to the model."""


class HandDetectorUnavailable(RuntimeError):
    """The hand detector failed to initialize or process the image."""


@dataclass(frozen=True)
class HandDetection:
    detected: bool


def validate_hand_image(image_bytes, detector=None):
    """Check the same center-square region used by inference, with EXIF rotation."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        crop = image.crop((left, top, left + side, top + side))
        crop.thumbnail((1024, 1024))
        rgb = np.asarray(crop)
    if detector is None:
        detector = _detect_hand
    try:
        detected = detector(rgb)
    except Exception as exc:
        raise HandDetectorUnavailable("Pemeriksaan tangan sedang tidak tersedia.") from exc
    if not detected:
        raise HandNotDetected(
            "Tangan tidak terdeteksi. Gunakan foto telapak tangan terbuka dan terlihat jelas."
        )
    return HandDetection(detected=True)


def _detect_hand(rgb):
    # Static-image mode resets tracking on each image; no hand data is saved.
    import mediapipe as mp
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        result = hands.process(rgb)
    return bool(result.multi_hand_landmarks)
