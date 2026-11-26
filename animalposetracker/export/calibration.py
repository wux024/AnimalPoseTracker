"""Calibration data for INT8 export.

Static quantization measures activation ranges from real inputs, so those inputs must
travel through exactly the same preprocessing as inference. Re-implementing the resize
here would shift the ranges, and the quantized model would lose accuracy for a reason
that is very hard to trace back.
"""

from pathlib import Path
from typing import Iterator, List

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def calibration_images(data_config_path, limit: int = 200) -> List[Path]:
    """First ``limit`` training images, in path order."""
    import yaml

    config_path = Path(data_config_path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = Path(payload.get("path") or ".").expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    train = payload.get("train")
    if isinstance(train, (list, tuple)):
        train = train[0] if train else None
    if not train:
        raise ValueError(f"dataset config {config_path} has no train split to calibrate on")

    directory = Path(str(train))
    if not directory.is_absolute():
        directory = root / directory
    paths = sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no calibration images under {directory}")
    return paths[:limit]


def calibrated_tensor(image_path, width: int, height: int) -> np.ndarray:
    """One preprocessed batch, ``(1, 3, H, W)`` float32, exactly as inference builds it."""
    import cv2

    from animalposetracker.preprocessing.live import preprocess_live_frame

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"calibration image is unreadable: {image_path}")
    tensor, _ = preprocess_live_frame(image, width, height)
    return tensor


def iter_tensors(paths, width: int, height: int) -> Iterator[np.ndarray]:
    for path in paths:
        yield calibrated_tensor(path, width, height)


__all__ = ["calibration_images", "calibrated_tensor", "iter_tensors"]
