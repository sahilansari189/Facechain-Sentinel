"""Face-region cropping helpers for secondary visual-search passes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def expanded_bbox(
    bbox: Sequence[float],
    image_shape: Sequence[int],
    margin: float = 0.35,
) -> tuple[int, int, int, int]:
    """Return a clipped bbox expanded around its center by ``margin``."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1, y1, x2, y2")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    x1 -= box_width * margin
    y1 -= box_height * margin
    x2 += box_width * margin
    y2 += box_height * margin
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


def crop_face(image: np.ndarray, bbox: Sequence[float], margin: float = 0.35) -> np.ndarray:
    """Crop an expanded face region, raising for an empty crop."""
    x1, y1, x2, y2 = expanded_bbox(bbox, image.shape, margin)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("face bbox produced an empty crop")
    return cropped.copy()