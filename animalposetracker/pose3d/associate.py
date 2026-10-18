"""Decide which instances are the same animal across synchronised cameras.

Synchronisation fixes *when* a frame was captured; it says nothing about
*which* animal is which.  With one animal per frame the detection order is
enough (``index``), but two animals swap places in that order as soon as one
camera drops or splits a detection, and triangulating the wrong pair still
produces a perfectly smooth-looking animal hanging in mid-air.

The identity question is answered with geometry instead of appearance: a
keypoint seen by one camera must lie on the epipolar line of its counterpart in
the other, so the pairing with the smallest Sampson distance is the only one
that is physically possible.  Every camera pair is matched one-to-one, then the
pairs are merged cheapest-first while a camera may contribute at most one
instance per group — a rig where one camera carries a larger residual still
keeps the other two paired instead of losing the frame.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .calibration import CameraRig
from .sync import fundamental_matrix, sampson_distance

#: Cost written into a cell no pair may take; the assignment skips those cells.
UNMATCHABLE = 1e9


def _pair_cost(
    first: np.ndarray,
    second: np.ndarray,
    fundamental: np.ndarray,
    *,
    confidence_threshold: float,
    min_points: int,
) -> float:
    """Median Sampson distance (px) over the keypoints both views are sure about.

    Returns ``inf`` when the two poses cannot be compared: different keypoint
    counts, too few jointly confident keypoints, or no finite distance.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b) or len(a) == 0:
        return float("inf")
    confidence_a = a[:, 2] if a.shape[1] > 2 else np.ones(len(a))
    confidence_b = b[:, 2] if b.shape[1] > 2 else np.ones(len(b))
    usable = (
        np.isfinite(a[:, :2]).all(axis=1)
        & np.isfinite(b[:, :2]).all(axis=1)
        & (confidence_a >= confidence_threshold)
        & (confidence_b >= confidence_threshold)
    )
    if int(usable.sum()) < min_points:
        return float("inf")
    distances = sampson_distance(a[usable, :2], b[usable, :2], fundamental)
    distances = distances[np.isfinite(distances)]
    if distances.size < min_points:
        return float("inf")
    return float(np.median(distances))


def associate_instances(
    poses_by_camera: Mapping[str, Sequence[np.ndarray]],
    rig: CameraRig,
    *,
    max_sampson_px: float = 25.0,
    confidence_threshold: float = 0.1,
    min_points: int = 4,
    min_views: int = 2,
) -> List[Dict[str, int]]:
    """Group per-camera instances into animals.

    ``poses_by_camera`` maps a camera to its instances for one frame, each an
    ``(K, 2|3)`` array of **undistorted** pixels (the third channel is
    confidence).  Cameras missing from the mapping, or holding no instances,
    simply take no part in that frame.

    Returns one mapping per animal, ``{camera: instance index}``, keeping only
    groups seen by at least ``min_views`` cameras.  Every camera appears at most
    once per group, and the list is ordered by the best (camera, instance) it
    holds, so a frame with a stable detection layout keeps a stable order.
    """
    if not np.isfinite(max_sampson_px) or max_sampson_px <= 0.0:
        raise ValueError("max_sampson_px must be finite and positive")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if int(min_points) < 4:
        raise ValueError("min_points must be at least 4 to constrain a pose")
    if int(min_views) < 2:
        raise ValueError("min_views must be at least 2")

    order = [camera for camera in rig.cameras if camera in poses_by_camera]
    counts = {camera: len(poses_by_camera[camera]) for camera in order}
    cameras = [camera for camera in order if counts[camera] > 0]
    if not cameras:
        return []

    fundamentals: Dict[Tuple[str, str], np.ndarray] = {}

    def fundamental(first: str, second: str) -> np.ndarray:
        key = (first, second)
        if key not in fundamentals:
            fundamentals[key] = fundamental_matrix(rig, first, second)
        return fundamentals[key]

    # Every camera pair is matched one-to-one (Hungarian) first; the surviving
    # pairs become edges that are merged cheapest-first afterwards.
    edges: List[Tuple[float, str, int, str, int]] = []
    for position, first in enumerate(cameras):
        for second in cameras[position + 1:]:
            cost = np.full((counts[first], counts[second]), UNMATCHABLE, dtype=np.float64)
            for row in range(counts[first]):
                for column in range(counts[second]):
                    value = _pair_cost(
                        poses_by_camera[first][row],
                        poses_by_camera[second][column],
                        fundamental(first, second),
                        confidence_threshold=confidence_threshold,
                        min_points=min_points,
                    )
                    if value <= max_sampson_px:
                        cost[row, column] = value
            for row, column in _assign(cost):
                if cost[row, column] < UNMATCHABLE:
                    edges.append((float(cost[row, column]), first, int(row), second, int(column)))

    # Merge cheapest first; refusing a merge that would give one camera two
    # instances keeps every group a valid multi-view observation of one animal.
    groups: List[Dict[str, int]] = []
    for _cost, first, first_index, second, second_index in sorted(edges):
        first_group = _group_of(groups, first, first_index)
        second_group = _group_of(groups, second, second_index)
        if first_group is None and second_group is None:
            groups.append({first: first_index, second: second_index})
        elif first_group is not None and second_group is None:
            groups[first_group][second] = second_index
        elif first_group is None and second_group is not None:
            groups[second_group][first] = first_index
        elif first_group != second_group:
            if set(groups[first_group]) & set(groups[second_group]):
                continue
            groups[first_group].update(groups[second_group])
            groups[second_group] = {}

    position_of = {camera: index for index, camera in enumerate(order)}
    kept = [group for group in groups if len(group) >= int(min_views)]
    kept.sort(key=lambda group: min((position_of[camera], index)
                                    for camera, index in group.items()))
    return [{camera: group[camera] for camera in order if camera in group} for group in kept]


def _group_of(groups: List[Dict[str, int]], camera: str, index: int):
    """Index of the group already holding ``(camera, index)``, if any."""
    for position, group in enumerate(groups):
        if group.get(camera) == index:
            return position
    return None


def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    """One-to-one assignment minimising the total cost, without scipy if it is absent."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is already a core dependency
        return _greedy_assign(cost)
    rows, columns = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), columns.tolist()))


def _greedy_assign(cost: np.ndarray) -> List[Tuple[int, int]]:  # pragma: no cover - fallback
    """Cheapest-first matching; identical to the Hungarian result for clear cases."""
    order = np.argsort(cost, axis=None, kind="stable")
    used_rows, used_columns, pairs = set(), set(), []
    for flat in order:
        row, column = divmod(int(flat), cost.shape[1])
        if cost[row, column] >= UNMATCHABLE:
            break
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        pairs.append((row, column))
    return pairs


__all__ = ["UNMATCHABLE", "associate_instances"]
