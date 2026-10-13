"""Letterbox geometry shared by the deployment and offline preprocessing paths.

Both paths fit an image into a fixed canvas with the same scale rule and pad with the
same grey, but they finish differently on purpose:

- the offline path (``letterbox_image``) rounds the resized size and pastes it at an
  integer, centred offset;
- the deployment path (``preprocess_live_frame``) keeps the scale fractional and builds
  an affine matrix, whose offset places the image at the bottom-right of the canvas.

This module shares the parts that must not drift -- the scale rule, the rounding rule,
the centred-padding rule and the pad value. The finishing differences stay with their
adapters; they are documented here so nobody "fixes" one side into the other without
measuring what the other side's consumers expect.
"""


PAD_VALUE = 114


def letterbox_scale(src_w: int, src_h: int, dst_w: int, dst_h: int) -> float:
    """The shared fit scale: ``min(dst_w / src_w, dst_h / src_h)``."""
    return min(dst_w / src_w, dst_h / src_h)


def letterbox_resized_size(src_w: int, src_h: int, scale: float):
    """The offline path's resized size: rounded, at least one pixel on each side."""
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def letterbox_padding(dst_w: int, dst_h: int, resized_w: int, resized_h: int):
    """The offline path's centred integer padding."""
    return (dst_w - resized_w) // 2, (dst_h - resized_h) // 2


__all__ = [
    "PAD_VALUE",
    "letterbox_scale",
    "letterbox_resized_size",
    "letterbox_padding",
]
