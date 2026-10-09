"""Assignment and observation-affinity utilities for the independent trackers."""

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .types import PoseDetection


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def keypoints_to_bbox(keypoints, confidence_threshold: float = 0.0, padding: float = 0.05):
    """Derive a loose XYXY extent from visible joints when a detector box is absent."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("keypoints must have shape (K, 2) or (K, 3)")
    if points.shape[1] == 3:
        visible = points[:, 2] > float(confidence_threshold)
    else:
        visible = np.ones((len(points),), dtype=bool)
    visible &= np.isfinite(points[:, :2]).all(axis=1)
    if not visible.any():
        raise ValueError("cannot derive a box because no keypoints are visible")
    xy = points[visible, :2]
    lower = xy.min(axis=0)
    upper = xy.max(axis=0)
    extent = np.maximum(upper - lower, 2.0)
    center = (upper + lower) * 0.5
    lower = center - extent * 0.5
    upper = center + extent * 0.5
    margin = extent * float(padding)
    lower -= margin
    upper += margin
    return np.asarray([lower[0], lower[1], upper[0], upper[1]], dtype=np.float32)


def _pose_parts(value):
    if hasattr(value, "keypoints"):
        points = value.keypoints
        box = getattr(value, "bbox_xyxy", None)
    else:
        points = value
        box = None
    if points is None:
        return None, box
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("keypoints must have shape (K, 2) or (K, 3)")
    return points, box


def pose_similarity(first, second, config: TrackerConfig, first_bbox=None, second_bbox=None) -> float:
    """Return confidence-weighted OKS over joints visible in both poses."""
    left, detected_left_box = _pose_parts(first)
    right, detected_right_box = _pose_parts(second)
    if left is None or right is None or left.shape[0] != right.shape[0]:
        return float("nan")
    left_conf = left[:, 2] if left.shape[1] == 3 else np.ones((len(left),), dtype=np.float32)
    right_conf = right[:, 2] if right.shape[1] == 3 else np.ones((len(right),), dtype=np.float32)
    visible = (
        (left_conf >= config.pose_keypoint_thresh)
        & (right_conf >= config.pose_keypoint_thresh)
        & np.isfinite(left[:, :2]).all(axis=1)
        & np.isfinite(right[:, :2]).all(axis=1)
    )
    if int(visible.sum()) < config.pose_min_common_keypoints:
        return float("nan")

    first_bbox = detected_left_box if first_bbox is None else first_bbox
    second_bbox = detected_right_box if second_bbox is None else second_bbox
    scales = []
    for box, points in ((first_bbox, left[visible, :2]), (second_bbox, right[visible, :2])):
        if box is not None:
            x1, y1, x2, y2 = map(float, box)
            area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
            scales.append(np.sqrt(area))
        else:
            extent = np.ptp(points, axis=0)
            scales.append(max(float(np.sqrt(max(extent[0] * extent[1], 1.0))), 1.0))
    scale = max(float(np.mean(scales)), 1.0)

    if config.kpt_oks_sigmas is None:
        sigmas = np.full((len(left),), 0.05, dtype=np.float32)
    else:
        sigmas = np.asarray(config.kpt_oks_sigmas, dtype=np.float32)
        if sigmas.shape != (len(left),):
            raise ValueError(
                f"kpt_oks_sigmas has {len(sigmas)} values but each pose has {len(left)} keypoints"
            )
    delta = np.linalg.norm(left[visible, :2] - right[visible, :2], axis=1)
    selected_sigmas = sigmas[visible]
    oks = np.exp(-0.5 * np.square(delta / np.maximum(2.0 * selected_sigmas * scale, 1e-6)))
    weights = np.minimum(left_conf[visible], right_conf[visible]).astype(np.float32)
    return float(np.average(oks, weights=np.maximum(weights, 1e-6)))


def pose_distance(first, second, config: TrackerConfig) -> float:
    """Return one minus OKS; invalid/incompatible poses return NaN."""
    similarity = pose_similarity(first, second, config)
    return float("nan") if not np.isfinite(similarity) else 1.0 - similarity


def cosine_distance(first, second) -> float:
    if first.embedding is None or second.embedding is None:
        return float("nan")
    left = np.asarray(first.embedding, dtype=np.float32)
    right = np.asarray(second.embedding, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("ReID embeddings in a matched pair must have the same dimension")
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def linear_assignment(cost_matrix: np.ndarray, max_cost: float):
    """Solve a rectangular assignment and return accepted pairs plus unmatched rows/cols."""
    costs = np.asarray(cost_matrix, dtype=np.float64)
    rows, cols = costs.shape if costs.ndim == 2 else (0, 0)
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    safe = np.nan_to_num(costs, nan=np.inf, posinf=np.inf, neginf=np.inf)
    threshold = float(max_cost) + 1e-7
    safe[safe > threshold] = threshold + 1e6
    assigned_rows, assigned_cols = linear_sum_assignment(safe)
    matches = []
    matched_rows, matched_cols = set(), set()
    for row, col in zip(assigned_rows.tolist(), assigned_cols.tolist()):
        if np.isfinite(costs[row, col]) and costs[row, col] <= threshold:
            matches.append((row, col))
            matched_rows.add(row)
            matched_cols.add(col)
    unmatched_rows = [row for row in range(rows) if row not in matched_rows]
    unmatched_cols = [col for col in range(cols) if col not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


def class_gate(cost_matrix: np.ndarray, tracks, detections):
    result = np.asarray(cost_matrix, dtype=np.float32).copy()
    for row, track in enumerate(tracks):
        for col, detection in enumerate(detections):
            if track.class_id != detection.class_id:
                result[row, col] = np.inf
    return result


__all__ = [
    "bbox_iou",
    "keypoints_to_bbox",
    "pose_similarity",
    "pose_distance",
    "cosine_distance",
    "linear_assignment",
    "class_gate",
]
