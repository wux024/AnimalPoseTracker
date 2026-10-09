"""Optional, standalone temporal filters for 2D and 3D pose sequences.

These functions never modify their input and are not called automatically by
prediction or tracking code. Inputs have time on axis 0, keypoints on the
penultimate axis, and coordinates/confidence on the final axis.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


_FILTERS = {"none", "median", "savgol"}


def _finite_runs(mask):
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return zip(starts.tolist(), ends.tolist())


def _fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or len(values) < 3:
        return values
    result = values.copy()
    missing = ~np.isfinite(result)
    for start, end in _finite_runs(missing):
        if end - start > max_gap or start == 0 or end == len(result):
            continue
        if np.isfinite(result[start - 1]) and np.isfinite(result[end]):
            result[start:end] = np.linspace(result[start - 1], result[end], end - start + 2)[1:-1]
    return result


def _filter_1d(
    values: np.ndarray,
    method: str,
    window_length: int,
    polyorder: int,
    max_gap: int,
) -> np.ndarray:
    result = _fill_short_gaps(np.asarray(values, dtype=np.float64), max_gap)
    if method == "none":
        return result

    for start, end in _finite_runs(np.isfinite(result)):
        segment = result[start:end]
        if method == "median":
            window = min(int(window_length), len(segment))
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                result[start:end] = median_filter(segment, size=window, mode="nearest")
        else:
            window = min(int(window_length), len(segment))
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                order = min(int(polyorder), window - 1)
                result[start:end] = savgol_filter(segment, window_length=window, polyorder=order, mode="interp")
    return result


def _filter_pose_sequence(
    poses,
    coordinate_dims: int,
    method: str = "median",
    window_length: int = 5,
    polyorder: int = 2,
    confidence_threshold: Optional[float] = None,
    max_gap: int = 0,
) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float32)
    if values.ndim < 3:
        raise ValueError("pose sequences must have shape (time, ..., keypoints, coordinates)")
    if values.shape[-1] not in (coordinate_dims, coordinate_dims + 1):
        raise ValueError(
            f"the last dimension must be {coordinate_dims} coordinates, optionally followed by confidence"
        )
    method = str(method).lower()
    if method not in _FILTERS:
        raise ValueError(f"method must be one of {sorted(_FILTERS)}")
    if int(window_length) < 1 or int(max_gap) < 0 or int(polyorder) < 0:
        raise ValueError("window_length must be positive and polyorder/max_gap non-negative")
    if confidence_threshold is not None and not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if confidence_threshold is not None and values.shape[-1] == coordinate_dims:
        raise ValueError("confidence_threshold requires a confidence channel in the input")
    if method in {"median", "savgol"} and int(window_length) < 3:
        raise ValueError("median and savgol filters require window_length of at least 3")

    result = np.ascontiguousarray(values.copy())
    coordinates = result[..., :coordinate_dims].copy()
    coordinates[~np.isfinite(coordinates)] = np.nan
    if result.shape[-1] == coordinate_dims + 1:
        confidence_values = result[..., coordinate_dims]
        finite_confidence = np.isfinite(confidence_values)
        invalid_confidence = finite_confidence & (
            (confidence_values < 0.0) | (confidence_values > 1.0)
        )
        if invalid_confidence.any():
            raise ValueError("confidence values must be in [0, 1] or non-finite for missing points")
    if result.shape[-1] == coordinate_dims + 1 and confidence_threshold is not None:
        confidence = result[..., coordinate_dims]
        coordinates[(~np.isfinite(confidence)) | (confidence < float(confidence_threshold))] = np.nan

    if result.shape[0] == 0:
        result[..., :coordinate_dims] = coordinates
        return result

    flattened = coordinates.reshape(result.shape[0], -1)
    filtered = np.empty_like(flattened, dtype=np.float64)
    for index in range(flattened.shape[1]):
        filtered[:, index] = _filter_1d(
            flattened[:, index], method, int(window_length), int(polyorder), int(max_gap)
        )
    result[..., :coordinate_dims] = filtered.reshape(coordinates.shape).astype(np.float32)
    return result


def filter_pose_2d(
    poses,
    method: str = "median",
    window_length: int = 5,
    polyorder: int = 2,
    confidence_threshold: Optional[float] = None,
    max_gap: int = 0,
) -> np.ndarray:
    """Filter a 2D pose sequence shaped ``(T, ..., K, 2|3)``.

    The optional third value is confidence/visibility. Low-confidence points
    can be masked before filtering; ``max_gap`` interpolates only bounded gaps.
    Confidence values are preserved unchanged.
    """
    return _filter_pose_sequence(
        poses, 2, method, window_length, polyorder, confidence_threshold, max_gap
    )


def filter_pose_3d(
    poses,
    method: str = "median",
    window_length: int = 5,
    polyorder: int = 2,
    confidence_threshold: Optional[float] = None,
    max_gap: int = 0,
) -> np.ndarray:
    """Filter a 3D pose sequence shaped ``(T, ..., K, 3|4)``.

    The optional fourth value is confidence. Coordinates are filtered per joint
    and axis, while confidence values and all leading identity axes are kept.
    """
    return _filter_pose_sequence(
        poses, 3, method, window_length, polyorder, confidence_threshold, max_gap
    )


__all__ = ["filter_pose_2d", "filter_pose_3d"]
