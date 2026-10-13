"""The shared pieces of every NMS this package runs.

Three callers suppress duplicates on purpose-built backends and stay there:
the training/offline chain goes through torchvision (with the torch greedy loop
below as its fallback), the deployment chain goes through ``cv2.dnn.NMSBoxes``,
and the SimCC validator suppresses on OKS. What is genuinely common lives here:
grouping, ordering, the OKS similarity, one greedy skeleton and the two index
engines -- each moved over bit-for-bit from where it used to live, so the callers
can share without any of them changing behaviour.

Nothing at module level imports torch: the deployment decoder imports this module
too, and it has to keep loading in an environment without torch. The torch engine
imports it lazily inside the function.
"""

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


def order_desc(scores: np.ndarray) -> np.ndarray:
    """Indices of ``scores``, highest first.

    Exactly the expression the deployment decoder used inline (``argsort`` keeps
    its own tie order; do not "improve" it to a stable sort without measuring).
    """
    return np.argsort(scores)[::-1]


def group_by_class(class_ids: np.ndarray, *, agnostic: bool) -> List[np.ndarray]:
    """Candidate indices grouped by class, or one group covering them all."""
    if agnostic:
        return [np.arange(len(class_ids), dtype=np.int64)]
    class_ids = np.asarray(class_ids)
    return [np.flatnonzero(class_ids == class_id) for class_id in np.unique(class_ids)]


def group_by_key(keys: Sequence[Any]) -> Dict[Any, List[int]]:
    """Positions grouped by a key, in first-seen order."""
    grouped: Dict[Any, List[int]] = {}
    for position, key in enumerate(keys):
        grouped.setdefault(key, []).append(position)
    return grouped


def nms_indices_numpy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    agnostic: bool,
    confidence_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """The deployment chain's suppression, moved from ``output_decoders``.

    ``cv2.dnn.NMSBoxes`` is called through the module attribute on purpose: the
    stream tests patch ``output_decoders.cv2.dnn.NMSBoxes``, which swaps the
    attribute on the shared ``cv2.dnn`` module and therefore reaches this call too.
    """
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)
    keep: List[int] = []
    for indices in group_by_class(class_ids, agnostic=agnostic):
        if not len(indices):
            continue
        boxes = boxes_xyxy[indices]
        xywh = np.column_stack(
            [
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ]
        )
        selected = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores[indices].astype(float).tolist(),
            float(confidence_threshold),
            float(iou_threshold),
        )
        if selected is not None:
            keep.extend(indices[np.asarray(selected, dtype=np.int64).reshape(-1)].tolist())
    if not keep:
        return np.empty((0,), dtype=np.int64)
    keep_array = np.asarray(keep, dtype=np.int64)
    return keep_array[order_desc(scores[keep_array])]


def nms_indices_torch(boxes, scores, threshold: float):
    """Greedy NMS for torch tensors, moved from ``nms.py`` unchanged.

    Ultralytics' ``TorchNMS.nms`` loop. Kept as a function rather than folded into
    ``nms_indices_numpy``: the two backends differ (tensor ops vs ``cv2.dnn``) and
    the callers on each side depend on their own engine's exact behaviour.
    """
    import torch  # noqa: PLC0415 -- module stays importable without torch

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(0, descending=True)
    keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
    keep_count = 0
    while order.numel() > 0:
        current = order[0]
        keep[keep_count] = current
        keep_count += 1
        if order.numel() == 1:
            break

        remaining = order[1:]
        xx1 = torch.maximum(x1[current], x1[remaining])
        yy1 = torch.maximum(y1[current], y1[remaining])
        xx2 = torch.minimum(x2[current], x2[remaining])
        yy2 = torch.minimum(y2[current], y2[remaining])
        intersection = (xx2 - xx1).clamp_(min=0) * (yy2 - yy1).clamp_(min=0)
        if intersection.sum() == 0:
            order = remaining
            continue
        overlap = intersection / (areas[current] + areas[remaining] - intersection)
        order = remaining[overlap <= threshold]

    return keep[:keep_count]


def greedy_suppress(
    order: Sequence[int],
    *,
    suppress,
) -> List[int]:
    """Greedy suppression skeleton: keep the first index, ask ``suppress`` for the rest.

    ``suppress(current, rest)`` returns one boolean per candidate in ``rest`` -- True
    means that candidate is suppressed by ``current`` and leaves the race. The order
    is used exactly as given; tie handling is the caller's business.
    """
    kept: List[int] = []
    remaining = list(order)
    while remaining:
        current = remaining[0]
        kept.append(current)
        rest = remaining[1:]
        if not rest:
            break
        drop = suppress(current, rest)
        remaining = [index for index, dropped in zip(rest, drop) if not dropped]
    return kept


def oks_similarity(keypoints_a, keypoints_b, area_a, area_b, variances) -> float:
    """OKS between two keypoint sets -- copied line-for-line from the validator.

    The expression (``np.spacing(1)`` floor, doubled variance term, mean over all
    keypoints) is what the SimCC validation path has always applied; keep it as is.
    """
    delta = (
        np.asarray(keypoints_b, dtype=np.float32)[:, :2]
        - np.asarray(keypoints_a, dtype=np.float32)[:, :2]
    )
    denominator = variances * ((float(area_a) + float(area_b)) * 0.5 + np.spacing(1)) * 2.0
    return float(np.exp(-np.sum(delta * delta, axis=1) / denominator).mean())


__all__ = [
    "order_desc",
    "group_by_class",
    "group_by_key",
    "nms_indices_numpy",
    "nms_indices_torch",
    "greedy_suppress",
    "oks_similarity",
]
