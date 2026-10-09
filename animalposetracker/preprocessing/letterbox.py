"""Letterbox image transforms used by offline prediction."""

import cv2
import numpy as np


def letterbox_image(image: np.ndarray, width: int, height: int):
    """Resize an image into a fixed canvas and return scale plus integer padding."""
    original_height, original_width = image.shape[:2]
    scale = min(width / original_width, height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, left, top


__all__ = ["letterbox_image"]
