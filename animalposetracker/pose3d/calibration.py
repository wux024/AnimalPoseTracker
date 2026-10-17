"""Load and validate a project's multi-view camera calibration.

The calibration lives in ``configs/cameras.yaml`` of a project created with the
3D option, and is written by hand: this module never imports a third-party
format, it only reads that file and rejects it with a precise message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .core import CameraCalibration


class CalibrationError(ValueError):
    """Raised when ``configs/cameras.yaml`` is missing, empty, or malformed."""


@dataclass(frozen=True)
class CameraRig:
    """The cameras of one capture session and how their pixels are distorted."""

    cameras: Dict[str, CameraCalibration]
    distortion: Dict[str, Tuple[str, np.ndarray]] = field(default_factory=dict)
    world_unit: str = "mm"

    def undistort(self, camera_id: str, pixels: np.ndarray) -> np.ndarray:
        """Map observed pixels to the undistorted pixel grid the core expects.

        ``pixels`` is ``(K, 2)`` or ``(K, 3)``; the third channel (confidence) is
        passed through untouched. Non-finite rows stay non-finite.
        """
        model, coefficients = self.distortion.get(camera_id, ("none", np.zeros(4)))
        points = np.asarray(pixels, dtype=np.float64)
        if model == "none" or not coefficients.any():
            return points.copy()
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - cv2 is a hard dependency
            raise CalibrationError("Undistorting pixels requires OpenCV (cv2)") from exc

        confidence = points[:, 2:] if points.shape[1] > 2 else None
        xy = points[:, :2]
        valid = np.isfinite(xy).all(axis=1)
        result = xy.copy()
        if valid.any():
            matrix = self.cameras[camera_id].intrinsic_matrix
            if model == "fisheye":
                fixed = cv2.fisheye.undistortPoints(
                    xy[valid].reshape(-1, 1, 2), matrix, coefficients.reshape(1, -1), P=matrix
                )
            else:
                fixed = cv2.undistortPoints(
                    xy[valid].reshape(-1, 1, 2), matrix, coefficients.reshape(1, -1), P=matrix
                )
            result[valid] = fixed.reshape(-1, 2)
        return np.concatenate([result, confidence], axis=1) if confidence is not None else result


def _require(mapping: Mapping[str, Any], key: str, camera: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise CalibrationError(f"Camera '{camera}' is missing the required field '{key}'")
    return mapping[key]


def _matrix(values: Any, rows: int, cols: int, camera: str, key: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != rows * cols:
        raise CalibrationError(
            f"Camera '{camera}' field '{key}' must hold {rows}x{cols} values, got {array.size}"
        )
    return array.reshape(rows, cols)


def load_camera_rig(path: Path) -> CameraRig:
    """Read a hand-written ``cameras.yaml`` and validate every camera."""
    import yaml

    source = Path(path)
    if not source.is_file():
        raise CalibrationError(f"Camera calibration file does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise CalibrationError(f"{source} must contain a mapping with a 'cameras' section")
    blocks = payload.get("cameras")
    if not isinstance(blocks, Mapping) or not blocks:
        raise CalibrationError(
            f"{source} has no cameras yet; fill in the 'cameras:' section (see the template)"
        )

    world_unit = str(payload.get("world_unit") or "mm").strip()
    if world_unit not in {"mm", "cm", "m"}:
        raise CalibrationError(f"world_unit must be one of mm, cm, m; got '{world_unit}'")

    cameras: Dict[str, CameraCalibration] = {}
    distortion: Dict[str, Tuple[str, np.ndarray]] = {}
    for name, block in blocks.items():
        camera = str(name).strip()
        if not isinstance(block, Mapping):
            raise CalibrationError(f"Camera '{camera}' must be a mapping of its calibration fields")
        intrinsic = _matrix(_require(block, "intrinsic", camera), 3, 3, camera, "intrinsic")
        rotation = _matrix(_require(block, "rotation", camera), 3, 3, camera, "rotation")
        translation = np.asarray(_require(block, "translation", camera), dtype=np.float64).reshape(-1)
        if translation.shape != (3,):
            raise CalibrationError(f"Camera '{camera}' field 'translation' must hold three values")
        image_size = block.get("image_size")
        if image_size is not None:
            width, height = (int(v) for v in image_size)
            image_size = (width, height)
        cameras[camera] = CameraCalibration(
            camera, intrinsic, rotation, translation,
            image_size=image_size, world_unit=world_unit,
        )

        settings = block.get("distortion") or {"model": "none", "coefficients": []}
        if not isinstance(settings, Mapping):
            raise CalibrationError(f"Camera '{camera}' field 'distortion' must be a mapping")
        model = str(settings.get("model") or "none").strip().lower()
        if model not in {"none", "brown", "fisheye"}:
            raise CalibrationError(
                f"Camera '{camera}' distortion model must be none, brown or fisheye; got '{model}'"
            )
        coefficients = np.asarray(settings.get("coefficients") or [], dtype=np.float64).reshape(-1)
        if model == "fisheye" and coefficients.size not in (4,):
            raise CalibrationError(f"Camera '{camera}' fisheye distortion needs 4 coefficients (k1..k4)")
        if model == "brown" and coefficients.size not in (4, 5, 8):
            raise CalibrationError(
                f"Camera '{camera}' brown distortion needs 4, 5 or 8 coefficients (k1,p1,p2,k2,k3)"
            )
        distortion[camera] = (model, coefficients)

    return CameraRig(cameras=cameras, distortion=distortion, world_unit=world_unit)


__all__ = ["CalibrationError", "CameraRig", "load_camera_rig"]
