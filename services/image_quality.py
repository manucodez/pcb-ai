"""
=========================================================
Input Image Quality Checks

Purpose
-------
A blurry, tiny, or badly exposed photo of a board will not
suddenly become detectable just because it was run through a
higher imgsz. Catching that up front and telling the user is
cheaper than a confusing "no defects found" and more useful
than a silent miss. This module runs a few cheap OpenCV checks
before the expensive YOLO + Gemini pass.

These are heuristics, not hard gates — the pipeline still runs
regardless. They only produce advisory warnings.

Author : Manjeet
=========================================================
"""

from __future__ import annotations

import cv2
import numpy as np

# Below this, small defects (mouse bites, spurs, pin-holes) are
# unlikely to survive resizing to the model's inference resolution.
MIN_RECOMMENDED_SIDE = 640

# Variance of the Laplacian is a standard cheap focus measure —
# lower means blurrier. This threshold is a rule of thumb tuned
# for close-up board photography, not an absolute physical unit.
BLUR_VARIANCE_THRESHOLD = 60.0

DARK_MEAN_THRESHOLD = 40.0
BRIGHT_MEAN_THRESHOLD = 235.0


def assess_quality(img_bgr: np.ndarray) -> dict:
    """Returns a dict with raw metrics plus a list of short,
    human-readable warning strings (empty if nothing stood out)."""

    warnings: list[str] = []

    if img_bgr is None or img_bgr.size == 0:
        return {"warnings": ["Image could not be read."], "ok": False}

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    short_side = min(h, w)
    if short_side < MIN_RECOMMENDED_SIDE:
        warnings.append(
            f"Image resolution is low ({w}×{h}px). Small defects may be "
            f"missed — {MIN_RECOMMENDED_SIDE}px on the short side or "
            "higher is recommended."
        )

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        warnings.append(
            "Image appears blurry or out of focus, which lowers detection "
            "reliability. Re-shoot with better focus/lighting if possible."
        )

    mean_brightness = float(gray.mean())
    if mean_brightness < DARK_MEAN_THRESHOLD:
        warnings.append("Image is very dark — consider improving lighting.")
    elif mean_brightness > BRIGHT_MEAN_THRESHOLD:
        warnings.append("Image is overexposed — consider reducing glare/flash.")

    return {
        "width": w,
        "height": h,
        "blur_variance": round(blur_variance, 1),
        "mean_brightness": round(mean_brightness, 1),
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }
