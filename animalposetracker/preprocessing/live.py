"""Input preprocessing for the existing live inference backend contract."""

import cv2
import numpy as np

from animalposetracker.preprocessing.letterbox_core import PAD_VALUE, letterbox_scale


def preprocess_live_frame(input_image, input_width: int, input_height: int, coreml: bool = False):
    """Apply the live engine's established affine transform and tensor layout."""
    img = input_image.copy()
    img_height, img_width = img.shape[:2]
    scale = letterbox_scale(img_width, img_height, input_width, input_height)
    ox = input_width - scale * img_width
    oy = input_height - scale * img_height
    matrix = np.array([[scale, 0, ox], [0, scale, oy]], dtype="float32")
    img = cv2.warpAffine(
        img,
        matrix,
        (input_width, input_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    inverse = cv2.invertAffineTransform(matrix)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if coreml:
        try:
            from PIL import Image

            img = Image.fromarray(img)
        except ImportError as exc:
            raise ImportError("Please install Pillow to use CoreML engine.") from exc
        return img, inverse
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, inverse


__all__ = ["preprocess_live_frame"]
