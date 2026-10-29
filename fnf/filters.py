"""Controlled beauty-filter treatments.

Every filter takes an RGB uint8 image and a strength in [0, 1] and returns an
RGB uint8 image. Strength 0 is always the identity, so the unfiltered image is
the control arm of the dose-response experiment.

Filters only touch skin pixels (soft mask), which is how phone beauty modes
behave, and keeps eyes, hair and background unchanged.
"""
import cv2
import numpy as np


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft skin mask in [0, 1] from YCrCb thresholds, feathered at the edges.

    The Cr/Cb box (133-173, 77-127) is a standard skin rule that works across
    light and dark skin because it ignores luminance (Y).
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    m = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    k = max(3, (min(rgb.shape[:2]) // 40) | 1)
    return cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)


def _blend(orig: np.ndarray, new: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None]
    out = orig.astype(np.float32) * (1 - m) + new.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Skin smoothing: edge-preserving bilateral filter blended in by strength."""
    if strength <= 0:
        return rgb.copy()
    d = int(5 + 10 * strength)
    sigma = 20 + 80 * strength
    filtered = cv2.bilateralFilter(rgb, d, sigma, sigma)
    return _blend(rgb, filtered, skin_mask(rgb) * strength)


def lighten(rgb: np.ndarray, strength: float, max_delta_l: float = 25.0) -> np.ndarray:
    """Skin lightening: raise CIELAB L* on skin by up to max_delta_l at strength 1.

    Also pulls a* slightly toward neutral, which is what most commercial
    whitening effects do (they desaturate redness as well as brighten).
    """
    if strength <= 0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab[..., 0] = np.clip(lab[..., 0] + max_delta_l * strength, 0, 100)
    lab[..., 1] = lab[..., 1] * (1 - 0.25 * strength)
    lit = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    lit = np.clip(lit * 255.0, 0, 255).astype(np.uint8)
    return _blend(rgb, lit, skin_mask(rgb))


def smooth_lighten(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Combined beauty mode: smoothing followed by lightening."""
    return lighten(smooth(rgb, strength), strength)


def jpeg(rgb: np.ndarray, quality: int = 85) -> np.ndarray:
    """Re-encode as JPEG, simulating the save a phone does after filtering."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


FILTERS = {
    "smooth": smooth,
    "lighten": lighten,
    "smooth_lighten": smooth_lighten,
}


def apply(rgb: np.ndarray, name: str, strength: float, jpeg_quality: int | None = 85) -> np.ndarray:
    """Apply a named filter, then a JPEG save so every arm (including the
    control at strength 0) goes through the same compression."""
    out = FILTERS[name](rgb, strength)
    return jpeg(out, jpeg_quality) if jpeg_quality else out
