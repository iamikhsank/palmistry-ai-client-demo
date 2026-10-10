"""Runtime helpers for camera/upload inference with the production checkpoint."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_SHA256 = "b2be1735907ba2b09ccf742675648617645f135e4924aaa555864a79e47b85ab"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_checkpoint() -> Path:
    """Rebuild the checkpoint from web-upload-safe repository parts."""
    target = Path(tempfile.gettempdir()) / "palm_resnet50_best.pth"
    if target.exists() and _sha256(target) == MODEL_SHA256:
        return target

    parts = sorted(MODEL_DIR.glob("palm_resnet50_best.pth.part-*"))
    if not parts:
        raise FileNotFoundError("Checkpoint parts are missing from the deployment.")

    with target.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)

    if _sha256(target) != MODEL_SHA256:
        target.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint integrity validation failed.")
    return target


class PalmImageProcessor:
    """Create the same BGR + Hessian four-channel format used for training."""

    def __init__(self, target_size: int = 1024) -> None:
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.sigmas = (1.2, 2.0, 3.5, 5.0)
        self.beta = 0.5
        self.c = 15.0

    def decode(self, uploaded: Any) -> np.ndarray:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        rgb = np.asarray(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def align_center(self, image_bgr: np.ndarray) -> np.ndarray:
        """Square center crop; camera guidance keeps the palm upright and centered."""
        height, width = image_bgr.shape[:2]
        side = min(height, width)
        x1 = max(0, (width - side) // 2)
        y1 = max(0, (height - side) // 2)
        crop = image_bgr[y1 : y1 + side, x1 : x1 + side]
        return cv2.resize(
            crop,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        enhanced = self.clahe.apply(lightness)
        return cv2.cvtColor(
            cv2.merge([enhanced, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR,
        )

    def ridge_map(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        max_response = np.zeros_like(gray, dtype=np.float32)

        for sigma in self.sigmas:
            blurred = ndimage.gaussian_filter(gray, sigma=sigma, mode="reflect")
            dy, dx = np.gradient(blurred)
            dyy, dyx = np.gradient(dy)
            dxy, dxx = np.gradient(dx)
            dxy = (dxy + dyx) / 2.0
            scale = sigma**2
            dxx, dxy, dyy = dxx * scale, dxy * scale, dyy * scale

            trace = dxx + dyy
            determinant = (dxx * dyy) - (dxy**2)
            discriminant = np.sqrt(np.maximum(0.0, trace**2 - 4.0 * determinant))
            lambda_1 = (trace - discriminant) / 2.0
            lambda_2 = (trace + discriminant) / 2.0
            swap = np.abs(lambda_1) > np.abs(lambda_2)
            small = np.where(swap, lambda_2, lambda_1)
            large = np.where(swap, lambda_1, lambda_2)
            condition = large > 0.0
            ratio_sq = (small / (large + 1e-6)) ** 2
            magnitude_sq = small**2 + large**2
            response = np.exp(-ratio_sq / (2.0 * self.beta**2)) * (
                1.0 - np.exp(-magnitude_sq / (2.0 * self.c**2))
            )
            vesselness = np.zeros_like(gray, dtype=np.float32)
            vesselness[condition] = response[condition]
            max_response = np.maximum(max_response, vesselness)

        minimum, maximum = float(max_response.min()), float(max_response.max())
        if maximum <= minimum:
            return np.zeros_like(gray, dtype=np.uint8)
        return ((max_response - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)

    def prepare(self, uploaded: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        raw_bgr = self.decode(uploaded)
        aligned = self.align_center(raw_bgr)
        clahe_bgr = self.enhance(aligned)
        ridge = self.ridge_map(clahe_bgr)
        tensor_4ch = np.dstack([clahe_bgr, ridge])

        gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
        quality = {
            "brightness": float(gray.mean()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        }
        return tensor_4ch, cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB), ridge, quality


def load_production_predictor():
    from modules.palm_dl_pipeline import PalmPredictor

    checkpoint = assemble_checkpoint()
    return PalmPredictor(
        model_weights_path=str(checkpoint),
        backbone="resnet50",
        device="cpu",
    )


def predict_image(predictor: Any, tensor_4ch: np.ndarray, threshold: float):
    return predictor.predict(
        tensor_4ch,
        stem="live_input",
        confidence_threshold=threshold,
        top_k_gradcam=0,
    )
