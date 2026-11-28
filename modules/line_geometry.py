"""Experimental ridge geometry: candidates only, never semantic line labels.

Inputs must already be a centered square ridge map. Pixel length is expressed
relative to the resized image width, not physical length. Crossings and noisy
textures can be counted as branches; no calibration or clinical claims.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _neighbors(mask):
    p = np.pad(mask, 1)
    return [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]


def thin(mask, max_iterations=256):
    """Zhang-Suen thinning with bounded iterations; no semantic inference."""
    image = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 2:
        raise ValueError("Mask must be two-dimensional")
    for _ in range(max_iterations):
        changed = False
        for phase in (0, 1):
            n = _neighbors(image)
            count = sum(x.astype(np.uint8) for x in n)
            transitions = sum((~n[i] & n[(i + 1) % 8]).astype(np.uint8)
                              for i in range(8))
            if phase == 0:
                keep1 = ~(n[0] & n[2] & n[4])
                keep2 = ~(n[2] & n[4] & n[6])
            else:
                keep1 = ~(n[0] & n[2] & n[6])
                keep2 = ~(n[0] & n[4] & n[6])
            remove = image & (count >= 2) & (count <= 6) & (transitions == 1) & keep1 & keep2
            if remove.any():
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def describe_mask(mask, min_pixels=15, max_candidates=5):
    """Compute geometric descriptors of connected skeleton components.

    `length_relative_width` is an approximate skeleton graph edge total divided
    by image width; forks, junctions, loops, and noise make it unlike a main-line
    length. Branch counts are clusters of junction pixels, not verified branches.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 3:
        raise ValueError("Mask must be a 2D image at least 3x3")
    skeleton = thin(mask)
    labels, count = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=int))
    results = []
    height, width = mask.shape
    for label in range(1, count + 1):
        component = labels == label
        ys, xs = np.nonzero(component)
        if len(xs) < min_pixels:
            continue
        n = _neighbors(component)
        degrees = sum(x.astype(np.uint8) for x in n)
        junctions, junction_count = ndimage.label(component & (degrees >= 3),
                                                  structure=np.ones((3, 3), dtype=int))
        edge_length = (np.count_nonzero(n[2] & component) + np.count_nonzero(n[4] & component)
                       + np.sqrt(2) * (np.count_nonzero(n[3] & component)
                                       + np.count_nonzero(n[5] & component)))
        results.append({"pixels": int(len(xs)),
                        "length_relative_width": float(edge_length / width),
                        "bbox_relative": (float(xs.min()/width), float(ys.min()/height),
                                          float((xs.max()+1)/width), float((ys.max()+1)/height)),
                        "junction_clusters": int(junction_count)})
    return sorted(results, key=lambda item: item["pixels"], reverse=True)[:max_candidates]


def palm_mask_from_landmarks(landmarks, size=256, margin_fraction=0.015):
    """Approximate palm-only polygon from normalized MediaPipe hand landmarks.

    The contour is approximate, not a segmentation or a line label. Landmarks
    must be from the same square crop as the ridge image, before resizing.
    """
    points = np.asarray(landmarks, dtype=float)
    if (points.ndim != 2 or points.shape[0] < 21 or points.shape[1] != 2
            or not np.isfinite(points).all() or size < 32
            or not 0 <= margin_fraction <= 0.1):
        raise ValueError("Invalid palm landmarks or mask parameters")
    # Refuse clipped hands rather than silently extrapolate the palm boundary.
    selected = points[[0, 1, 5, 9, 13, 17]]
    if np.any(selected < 0) or np.any(selected > 1):
        raise ValueError("Palm landmarks extend beyond the analyzed image")
    polygon = [(int(round(x * (size - 1))), int(round(y * (size - 1))))
               for x, y in selected]
    canvas = Image.new("1", (size, size), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=1)
    mask = np.asarray(canvas, dtype=bool)
    margin = int(round(size * margin_fraction))
    if margin:
        mask = ndimage.binary_erosion(mask, iterations=margin)
    if mask.sum() < size * size * 0.01:
        raise ValueError("Detected palm area is too small")
    return mask


def detect_palm_landmarks(rgb):
    """Detect landmarks in the same square image used for model inference."""
    import mediapipe as mp
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1]:
        raise ValueError("Expected a square RGB inference crop")
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                   min_detection_confidence=0.5) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    return np.array([(p.x, p.y) for p in result.multi_hand_landmarks[0].landmark])


def analyze_ridge_candidates(ridge, size=256, threshold_percentile=85, palm_mask=None):
    """Extract thresholded candidates; output is exploratory, not line recognition."""
    ridge = np.asarray(ridge)
    if ridge.ndim != 2 or not np.isfinite(ridge).all():
        raise ValueError("Ridge map must be finite and 2D")
    if not 0 <= threshold_percentile <= 100 or size < 32:
        raise ValueError("Invalid geometry parameters")
    ridge = np.asarray(Image.fromarray(np.clip(ridge, 0, 255).astype(np.uint8))
                       .resize((size, size), Image.Resampling.BILINEAR))
    if np.ptp(ridge) == 0:
        return []
    if palm_mask is not None:
        palm_mask = np.asarray(palm_mask, dtype=bool)
        if palm_mask.shape != ridge.shape or not palm_mask.any():
            raise ValueError("Palm mask must be nonempty and match the resized ridge")
    else:
        palm_mask = np.ones_like(ridge, dtype=bool)
    threshold = max(25.0, float(np.percentile(ridge[palm_mask], threshold_percentile)))
    mask = (ridge > threshold) & palm_mask
    return describe_mask(mask)
