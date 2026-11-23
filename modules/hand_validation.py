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


def estimate_sharpness(image_bytes):
    """Heuristic on the unenhanced center crop, not a validated quality score."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        crop = image.crop((left, top, left + side, top + side)).resize((512, 512))
        gray = np.asarray(crop, dtype=np.float32)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    return float((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2)


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
            "Tangan tidak terdeteksi. Foto hanya telapak tangan terbuka dengan seluruh jari "
            "terlihat, posisikan di tengah, dan gunakan cahaya merata. "
            "Jika foto terpotong atau miring, ambil ulang dari jarak sedikit lebih jauh."
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
