"""Skin tone via the Individual Typology Angle (ITA).

ITA = arctan((L* - 50) / b*) * 180 / pi, computed on CIELAB skin pixels.
Higher ITA means lighter skin. Bands follow Chardon et al. (1991), the
convention used in dermatology and in face-fairness work.
"""
import cv2
import numpy as np

from .filters import skin_mask

# (lower bound, name); an ITA falls in the first band whose bound it exceeds.
ITA_BANDS = [
    (55, "very_light"),
    (41, "light"),
    (28, "intermediate"),
    (10, "tan"),
    (-30, "brown"),
    (-np.inf, "dark"),
]
BAND_ORDER = [name for _, name in ITA_BANDS]


def ita(rgb: np.ndarray, min_pixels: int = 50) -> float:
    """Median ITA over skin pixels. Returns NaN if too few skin pixels."""
    mask = skin_mask(rgb) > 0.5
    # Use the central face region only, to avoid neck/background bleed.
    h, w = mask.shape
    centre = np.zeros_like(mask)
    centre[int(0.2 * h):int(0.8 * h), int(0.25 * w):int(0.75 * w)] = True
    mask &= centre
    if mask.sum() < min_pixels:
        return float("nan")
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L = lab[..., 0][mask]
    b = lab[..., 2][mask]
    # arctan2 keeps the sign right when b* is near zero.
    angles = np.degrees(np.arctan2(L - 50.0, b))
    return float(np.median(angles))


def band(ita_value: float) -> str:
    if np.isnan(ita_value):
        return "unknown"
    for lower, name in ITA_BANDS:
        if ita_value > lower:
            return name
    return "dark"
