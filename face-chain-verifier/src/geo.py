"""Conservative, local scene metadata extraction.

This module intentionally reports observable image properties only. It does not
claim a precise location or infer identity from scene context.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def analyze_scene(image: np.ndarray) -> Dict[str, Any]:
    """Return lightweight scene cues suitable for analyst review."""
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    pixels = image.astype(np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    height, width = image.shape[:2]
    aspect = width / max(height, 1)
    return {
        "dimensions": {"width": int(width), "height": int(height)},
        "aspect_ratio": round(aspect, 4),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "lighting": "low" if brightness < 70 else "high" if brightness > 185 else "moderate",
        "composition": "landscape" if aspect > 1.2 else "portrait" if aspect < 0.8 else "square",
    }