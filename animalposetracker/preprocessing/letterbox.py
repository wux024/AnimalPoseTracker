"""Letterbox image transforms used by offline prediction."""

import cv2
import numpy as np

from animalposetracker.preprocessing.letterbox_core import (
    PAD_VALUE,
    letterbox_padding,
    letterbox_resized_size,
    letterbox_scale,
)


def letterbox_image(image: np.ndarray, width: int, height: int):
    """Resize an image into a fixed canvas and return scale plus integer padding."""
    original_height, original_width = image.shape[:2]
    scale = letterbox_scale(original_width, original_height, width, height)
    resized_width, resized_height = letterbox_resized_size(original_width, original_height, scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left, top = letterbox_padding(width, height, resized_width, resized_height)
    canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, left, top


__all__ = ["letterbox_image"]
