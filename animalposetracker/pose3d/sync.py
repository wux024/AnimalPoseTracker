"""Estimate the frame offset between cameras from their tracked 2D keypoints.

Hardware-synchronised cameras need no offset, but files recorded separately often start a
few frames apart, and pairing frame i with frame i silently biases every 3D result. The
check here needs no calibration board: for a candidate offset the two views' keypoints must
satisfy the epipolar constraint (the rig already gives the fundamental matrix), so the
offset that minimises the median Sampson distance is the one that actually aligns the views.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .calibration import CameraRig


def fundamental_matrix(rig: CameraRig, first: str, second: str) -> np.ndarray:
    """The fundamental matrix mapping pixels in ``first`` to epipolar lines in ``second``."""
    a = rig.cameras[first]
    b = rig.cameras[second]
    rotation = b.rotation_matrix @ a.rotation_matrix.T
    translation = b.translation_vector - rotation @ a.translation_vector
    skew = np.array([[0.0, -translation[2], translation[1]],
                     [translation[2], 0.0, -translation[0]],
                     [-translation[1], translation[0], 0.0]])
    essential = skew @ rotation
    return np.linalg.inv(b.intrinsic_matrix).T @ essential @ np.linalg.inv(a.intrinsic_matrix)


def sampson_distance(points_a: np.ndarray, points_b: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    """Sampson distance (px) between matched points and their epipolar lines."""
    ones = np.ones((len(points_a), 1))
    homogeneous_a = np.hstack([points_a, ones])
    homogeneous_b = np.hstack([points_b, ones])
    lines_b = homogeneous_a @ fundamental.T          # epipolar lines in image b
    lines_a = homogeneous_b @ fundamental            # epipolar lines in image a
    numerator = np.sum(homogeneous_b * lines_b, axis=1) ** 2
    denominator = (lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2 +
                   lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _paired_points(series_a: Sequence[np.ndarray], series_b: Sequence[np.ndarray],
                   offset: int) -> tuple:
    """Pair frame i of view a with frame i + offset of view b."""
    pairs_a, pairs_b = [], []
    for index in range(len(series_a)):
        other = index + offset
        if 0 <= other < len(series_b):
            pairs_a.append(np.asarray(series_a[index], dtype=np.float64)[:, :2])
            pairs_b.append(np.asarray(series_b[other], dtype=np.float64)[:, :2])
    if not pairs_a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(pairs_a), np.vstack(pairs_b)


def estimate_frame_offsets(
    rig: CameraRig,
    series: Mapping[str, Sequence[np.ndarray]],
    *,
    max_offset: int = 60,
    metric: str = "median",
) -> Dict[str, object]:
    """Sweep the frame offset of every camera against the anchor and pick the minimum.

    ``series`` holds, per camera, one ``(K, 3)`` array of undistorted pixels per frame.
    The anchor is the first camera; the returned ``offsets`` map every other camera to the
    number of frames its data lags (positive) or leads (negative) the anchor.
    """
    camera_ids = list(rig.cameras)
    if len(camera_ids) < 2:
        raise ValueError("An offset needs at least two cameras")
    anchor = camera_ids[0]
    if anchor not in series or len(series[anchor]) == 0:
        raise ValueError(f"No 2D series for the anchor camera '{anchor}'")

    report: Dict[str, object] = {"anchor": anchor, "max_offset": int(max_offset),
                                 "offsets": {}, "curves": {}}
    for camera in camera_ids[1:]:
        if camera not in series or not len(series[camera]):
            continue
        fundamental = fundamental_matrix(rig, anchor, camera)
        scores = []
        for offset in range(-int(max_offset), int(max_offset) + 1):
            points_a, points_b = _paired_points(series[anchor], series[camera], offset)
            if len(points_a) < 8:
                scores.append((offset, float("inf")))
                continue
            distances = sampson_distance(points_a, points_b, fundamental)
            distances = distances[np.isfinite(distances)]
            if distances.size < 8:
                scores.append((offset, float("inf")))
                continue
            value = float(np.median(distances) if metric == "median" else np.mean(distances))
            scores.append((offset, value))
        finite = [(offset, value) for offset, value in scores if np.isfinite(value)]
        if not finite:
            continue
        best_offset, best_value = min(finite, key=lambda item: item[1])
        zero_value = next((value for offset, value in scores if offset == 0), float("inf"))
        report["offsets"][camera] = int(best_offset)
        report["curves"][camera] = {
            "best_offset": int(best_offset),
            "best_median_px": round(best_value, 3),
            "zero_offset_median_px": round(zero_value, 3) if np.isfinite(zero_value) else None,
            "confident": bool(best_value < 0.6 * zero_value) if np.isfinite(zero_value) and zero_value > 0 else True,
        }
    return report


__all__ = ["fundamental_matrix", "sampson_distance", "estimate_frame_offsets"]
