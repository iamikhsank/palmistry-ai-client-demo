"""Contrast normalization for experimental palm-line detection.

The output highlights local dark structures, not identified palm lines.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def enhance_ridge(image, mask=None, radius=7, min_dynamic_range=3.0):
    """Return a normalized black-hat response, or None for insufficient signal.

    Operates on the full, unwarped square image to keep image, landmarks and
    overlays in the same coordinate system. Thresholds are experimental.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1]:
        raise ValueError('Expected a square RGB image')
    if rgb.size == 0 or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError('Expected finite RGB pixels between 0 and 255')
    if not isinstance(radius, int) or radius < 1:
        raise ValueError('radius must be a positive integer')
    gray = np.asarray(Image.fromarray(rgb.astype(np.uint8), 'RGB').convert('L'), dtype=np.uint8)
    axis = np.arange(-radius, radius + 1)
    footprint = (axis[:, None] ** 2 + axis[None, :] ** 2) <= radius ** 2
    response = ndimage.black_tophat(gray, footprint=footprint).astype(float)
    if mask is None:
        select = np.ones(gray.shape, dtype=bool)
    else:
        select = np.asarray(mask, dtype=bool)
        if select.shape != gray.shape or not select.any():
            raise ValueError('Mask must match image and contain foreground')
    lo, hi = np.percentile(response[select], [1, 99])
    if hi - lo < min_dynamic_range:
        return None
    out = np.clip((response - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    out[~select] = 0
    return out
