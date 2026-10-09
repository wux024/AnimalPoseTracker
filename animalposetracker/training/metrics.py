"""Pose metrics and COCO keypoint evaluation utilities."""

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes, returning a ``(len(boxes1), len(boxes2))`` matrix."""
    if not boxes1.numel() or not boxes2.numel():
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_(min=0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_(min=0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_(min=0).prod(dim=-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + eps)


def keypoint_oks(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    area: torch.Tensor,
    kpt_oks_sigmas: Sequence[float],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Pairwise OKS using visible GT keypoints and the reference ``0.53 * box area``."""
    if not ground_truth.numel() or not prediction.numel():
        return ground_truth.new_zeros((ground_truth.shape[0], prediction.shape[0]))
    squared_distance = (
        ground_truth[:, None, :, 0] - prediction[None, :, :, 0]
    ).square() + (
        ground_truth[:, None, :, 1] - prediction[None, :, :, 1]
    ).square()
    visible = ground_truth[..., 2] != 0
    sigma = torch.as_tensor(
        kpt_oks_sigmas, device=ground_truth.device, dtype=ground_truth.dtype
    )
    error = squared_distance / (
        (2.0 * sigma).square()[None, None, :] * (area[:, None, None] + eps) * 2.0
    )
    return (torch.exp(-error) * visible[:, None, :]).sum(dim=-1) / (
        visible.sum(dim=-1, keepdim=True) + eps
    )


def match_predictions(
    similarity: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_classes: torch.Tensor,
    thresholds: torch.Tensor = IOU_THRESHOLDS,
) -> torch.Tensor:
    """Greedily match same-class predictions and targets at each metric threshold."""
    correct = np.zeros((prediction_classes.shape[0], thresholds.numel()), dtype=bool)
    if not prediction_classes.numel() or not target_classes.numel():
        return torch.from_numpy(correct)

    overlaps = similarity.detach().float().cpu().numpy().copy()
    pred_cls = prediction_classes.detach().cpu().numpy()
    target_cls = target_classes.detach().cpu().numpy()
    overlaps *= target_cls[:, None] == pred_cls[None, :]
    for threshold_index, threshold in enumerate(thresholds.tolist()):
        matches = np.array(np.nonzero(overlaps >= threshold)).T
        if matches.shape[0] > 1:
            matches = matches[overlaps[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        if matches.shape[0]:
            correct[matches[:, 1].astype(int), threshold_index] = True
    return torch.from_numpy(correct)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Integrate the precision envelope with the COCO-style 101 point rule."""
    final_recall = recall[-1] if recall.size else 1.0
    modified_recall = np.concatenate(([0.0], recall, [final_recall], [1.0]))
    modified_precision = np.concatenate(([1.0], precision, [0.0], [0.0]))
    modified_precision = np.flip(np.maximum.accumulate(np.flip(modified_precision)))
    points = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(points, modified_recall, modified_precision), points))


def _smooth(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Smooth a curve with the moving-average window used to select max-F1 P/R."""
    window = round(values.size * fraction * 2) // 2 + 1
    padding = np.ones(window // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]))
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def summarize_ap(
    true_positives: Sequence[np.ndarray],
    confidences: Sequence[np.ndarray],
    prediction_classes: Sequence[np.ndarray],
    target_classes: Sequence[np.ndarray],
) -> Tuple[float, float, float, float]:
    """Return mean precision, recall, AP50 and AP50-95 as in the reference metrics."""
    tp = np.concatenate(true_positives, axis=0) if true_positives else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(confidences) if confidences else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(prediction_classes) if prediction_classes else np.zeros((0,), dtype=np.int64)
    target_cls = np.concatenate(target_classes) if target_classes else np.zeros((0,), dtype=np.int64)
    classes, target_counts = np.unique(target_cls, return_counts=True)
    if classes.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    x = np.linspace(0.0, 1.0, 1000)
    precision_curves = np.zeros((classes.size, x.size), dtype=np.float64)
    recall_curves = np.zeros_like(precision_curves)
    ap = np.zeros((classes.size, 10), dtype=np.float64)

    for class_index, class_id in enumerate(classes):
        selected = pred_cls == class_id
        number_of_labels = target_counts[class_index]
        if not selected.any() or number_of_labels == 0:
            continue
        class_confidence = conf[selected]
        class_tp = tp[selected].astype(np.float64)
        false_positives = (1.0 - class_tp).cumsum(axis=0)
        true_positives_cumulative = class_tp.cumsum(axis=0)
        recall = true_positives_cumulative / (number_of_labels + 1e-16)
        precision = true_positives_cumulative / (
            true_positives_cumulative + false_positives
        )
        recall_curves[class_index] = np.interp(
            -x, -class_confidence, recall[:, 0], left=0.0
        )
        precision_curves[class_index] = np.interp(
            -x, -class_confidence, precision[:, 0], left=1.0
        )
        for threshold_index in range(10):
            ap[class_index, threshold_index] = _compute_ap(
                recall[:, threshold_index], precision[:, threshold_index]
            )

    f1_curve = 2.0 * precision_curves * recall_curves / (
        precision_curves + recall_curves + 1e-16
    )


def _coco_keypoint_api():
    """Return the COCO API, preferring the extended API used by MMPose."""
    try:
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
        return COCO, COCOeval
    except ImportError:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            return COCO, COCOeval
        except ImportError as exc:
            raise ImportError(
                "COCO keypoint validation requires xtcocotools or pycocotools; "
                "install AnimalPoseTracker with its training extra."
            ) from exc


def build_coco_ground_truth(dataset):
    """Build the COCO ground truth and ID mappings for a validation dataset.

    Native COCO annotations are kept intact so their category IDs, instance areas,
    and image IDs retain their standard meaning. YOLO pose labels are converted to
    an in-memory COCO dataset with stable IDs for the validation images.
    """
    COCO, _COCOeval = _coco_keypoint_api()
    image_ids_by_path = dict(getattr(dataset, "image_id_by_path", {}))
    category_ids_by_class = dict(getattr(dataset, "category_id_by_class", {}))
    annotation_path = getattr(dataset, "annotation_path", None)
    if annotation_path is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(annotation_path))
        return coco_gt, image_ids_by_path, category_ids_by_class

    image_records = []
    annotations = []
    for image_index, image_path in enumerate(dataset.image_paths):
        image_path = Path(image_path).resolve()
        image_id = image_ids_by_path.get(str(image_path), image_index + 1)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read validation image for COCO evaluation: {image_path}")
        image_height, image_width = image.shape[:2]
        image_records.append({
            "id": int(image_id),
            "file_name": image_path.name,
            "width": int(image_width),
            "height": int(image_height),
        })
        classes, boxes, keypoints = dataset._read_targets(image_path, image_width, image_height)
        for class_id, box, points in zip(classes, boxes, keypoints):
            center_x, center_y, box_width, box_height = map(float, box)
            x = (center_x - box_width * 0.5) * image_width
            y = (center_y - box_height * 0.5) * image_height
            width = box_width * image_width
            height = box_height * image_height
            if dataset.keypoint_dimensions == 3:
                visibility = points[:, 2]
            else:
                visibility = np.ones((dataset.num_keypoints,), dtype=np.float32)
            keypoints_coco = np.empty((dataset.num_keypoints, 3), dtype=np.float32)
            keypoints_coco[:, :2] = points[:, :2] * np.asarray(
                [image_width, image_height], dtype=np.float32
            )
            keypoints_coco[:, 2] = visibility
            category_id = category_ids_by_class.get(int(class_id), int(class_id) + 1)
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": int(image_id),
                "category_id": int(category_id),
                "keypoints": keypoints_coco.reshape(-1).tolist(),
                "num_keypoints": int(np.count_nonzero(visibility > 0)),
                "bbox": [x, y, width, height],
                "area": float(max(width, 0.0) * max(height, 0.0)),
                "iscrowd": 0,
            })

    keypoint_names = dataset.data_config.get("keypoint_names") or [
        str(index) for index in range(dataset.num_keypoints)
    ]
    categories = []
    for class_id, class_name in enumerate(dataset.source_class_names):
        categories.append({
            "id": int(category_ids_by_class.get(class_id, class_id + 1)),
            "name": str(class_name),
            "keypoints": list(keypoint_names),
            "skeleton": [
                [int(edge[0]) + 1, int(edge[1]) + 1]
                for edge in (dataset.skeleton or [])
                if len(edge) == 2
            ],
        })
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "AnimalPoseTracker validation set"},
        "images": image_records,
        "annotations": annotations,
        "categories": categories,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    return coco_gt, image_ids_by_path, category_ids_by_class


def evaluate_coco_keypoints(
    coco_gt,
    detections: Sequence[Dict[str, Any]],
    kpt_oks_sigmas: Sequence[float],
    image_ids: Optional[Sequence[int]] = None,
    category_ids: Optional[Sequence[int]] = None,
    use_categories: bool = True,
    max_detections: int = 20,
    return_matches: bool = False,
):
    """Evaluate keypoint predictions with COCOeval and dataset-specific OKS sigmas.

    When requested, also return the ground-truth/detection pairs COCOeval matched
    at OKS 0.50. These pairs support localization diagnostics such as PCK/AUC/EPE.
    """
    COCO, COCOeval = _coco_keypoint_api()
    sigma_values = np.asarray(kpt_oks_sigmas, dtype=np.float32).reshape(-1)
    if not sigma_values.size or not np.isfinite(sigma_values).all() or np.any(sigma_values <= 0):
        raise ValueError("COCO keypoint evaluation needs one finite positive OKS sigma per keypoint")
    with contextlib.redirect_stdout(io.StringIO()):
        if detections:
            coco_dt = coco_gt.loadRes(list(detections))
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "info": coco_gt.dataset.get("info", {}),
                "images": list(coco_gt.dataset.get("images", [])),
                "categories": list(coco_gt.dataset.get("categories", [])),
                "annotations": [],
            }
            coco_dt.createIndex()

        evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
        # COCOeval's built-in vector is the human COCO-17 vector. Always replace it.
        evaluator.params.kpt_oks_sigmas = sigma_values
        evaluator.params.maxDets = [int(max_detections)]
        if image_ids is not None:
            evaluator.params.imgIds = sorted({int(image_id) for image_id in image_ids})
        if category_ids is not None:
            evaluator.params.catIds = sorted({int(category_id) for category_id in category_ids})
        evaluator.params.useCats = int(bool(use_categories))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = np.asarray(evaluator.stats, dtype=np.float64)
    stats = np.where(np.isfinite(stats) & (stats >= 0), stats, 0.0)
    metrics = {
        "coco/AP": float(stats[0]),
        "coco/AP50": float(stats[1]),
        "coco/AP75": float(stats[2]),
        "coco/AR": float(stats[5]),
    }
    if not return_matches:
        return metrics

    matched_pairs = []
    all_area_range = np.asarray(evaluator.params.areaRng[0], dtype=np.float64)
    for eval_image in evaluator.evalImgs or []:
        if eval_image is None or not np.allclose(
            np.asarray(eval_image.get("aRng", []), dtype=np.float64), all_area_range
        ):
            continue
        dt_ids = eval_image.get("dtIds", [])
        gt_ids = np.asarray(eval_image.get("dtMatches", []))[0]
        ignored = np.asarray(eval_image.get("dtIgnore", []))[0]
        for detection_id, ground_truth_id, is_ignored in zip(dt_ids, gt_ids, ignored):
            if int(ground_truth_id) <= 0 or bool(is_ignored):
                continue
            matched_pairs.append((
                coco_gt.anns[int(ground_truth_id)],
                coco_dt.anns[int(detection_id)],
            ))
    return metrics, matched_pairs


def keypoint_error_metrics(
    errors: np.ndarray,
    visible: np.ndarray,
    bbox_sizes: np.ndarray,
    pck_threshold: float = 0.05,
    auc_norm_factor: float = 30.0,
    auc_thresholds: int = 20,
) -> Dict[str, float]:
    """Compute MMPose-style PCK, AUC and EPE from per-instance point errors."""
    error_values = np.asarray(errors, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    bbox_sizes = np.maximum(np.asarray(bbox_sizes, dtype=np.float32).reshape(-1), 1.0)
    if error_values.size == 0 or visible.size == 0:
        return {"PCK": 0.0, "AUC": 0.0, "EPE": 0.0}
    if error_values.shape != visible.shape or error_values.shape[0] != bbox_sizes.shape[0]:
        raise ValueError("Keypoint errors, visibility masks and bbox sizes must align")
    normalized = error_values / bbox_sizes[:, None]
    per_keypoint_pck = []
    for keypoint_index in range(normalized.shape[1]):
        valid = visible[:, keypoint_index]
        if valid.any():
            per_keypoint_pck.append(
                float((normalized[valid, keypoint_index] < float(pck_threshold)).mean())
            )

    normalized_for_auc = error_values / float(auc_norm_factor)
    auc_values = []
    for threshold_index in range(int(auc_thresholds)):
        threshold = threshold_index / float(auc_thresholds)
        per_keypoint = []
        for keypoint_index in range(normalized_for_auc.shape[1]):
            valid = visible[:, keypoint_index]
            if valid.any():
                per_keypoint.append(
                    float((normalized_for_auc[valid, keypoint_index] < threshold).mean())
                )
        auc_values.append(float(np.mean(per_keypoint)) if per_keypoint else 0.0)
    return {
        "PCK": float(np.mean(per_keypoint_pck)) if per_keypoint_pck else 0.0,
        "AUC": float(np.mean(auc_values)),
        "EPE": float(error_values[visible].mean()) if visible.any() else 0.0,
    }


def keypoint_error_metrics_from_coco_matches(
    matched_pairs,
    pck_threshold: float = 0.05,
    auc_norm_factor: float = 30.0,
    auc_thresholds: int = 20,
) -> Dict[str, float]:
    """Compute PCK/AUC/EPE over instances matched by COCOeval at OKS 0.50."""
    errors = []
    visibility = []
    bbox_sizes = []
    for ground_truth, detection in matched_pairs:
        gt_points = np.asarray(ground_truth["keypoints"], dtype=np.float32).reshape(-1, 3)
        pred_points = np.asarray(detection["keypoints"], dtype=np.float32).reshape(-1, 3)
        if pred_points.shape[0] != gt_points.shape[0]:
            raise ValueError("Matched COCO keypoint predictions have a different keypoint count")
        errors.append(np.linalg.norm(pred_points[:, :2] - gt_points[:, :2], axis=-1))
        visibility.append(gt_points[:, 2] > 0)
        bbox = ground_truth.get("bbox", (0.0, 0.0, 1.0, 1.0))
        bbox_sizes.append(max(float(bbox[2]), float(bbox[3]), 1.0))
    if not errors:
        return {"PCK": 0.0, "AUC": 0.0, "EPE": 0.0}
    return keypoint_error_metrics(
        np.stack(errors),
        np.stack(visibility),
        np.asarray(bbox_sizes),
        pck_threshold=pck_threshold,
        auc_norm_factor=auc_norm_factor,
        auc_thresholds=auc_thresholds,
    )
    best_index = int(_smooth(f1_curve.mean(axis=0)).argmax())
    return (
        float(precision_curves[:, best_index].mean()),
        float(recall_curves[:, best_index].mean()),
        float(ap[:, 0].mean()),
        float(ap.mean()),
    )


def update_confusion_matrix(
    matrix: np.ndarray,
    box_similarity: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_classes: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    """Accumulate class-agnostic IoU matches; the final row/column is background."""
    number_of_classes = matrix.shape[0] - 1
    pred_cls = prediction_classes.detach().cpu().numpy().astype(np.int64, copy=False)
    target_cls = target_classes.detach().cpu().numpy().astype(np.int64, copy=False)
    overlaps = box_similarity.detach().float().cpu().numpy()
    matches = np.array(np.nonzero(overlaps >= threshold)).T
    if matches.shape[0] > 1:
        matches = matches[overlaps[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    matched_targets = set()
    matched_predictions = set()
    for target_index, prediction_index in matches:
        target_class = int(target_cls[target_index])
        prediction_class = int(pred_cls[prediction_index])
        if 0 <= target_class < number_of_classes and 0 <= prediction_class < number_of_classes:
            matrix[target_class, prediction_class] += 1
        matched_targets.add(int(target_index))
        matched_predictions.add(int(prediction_index))
    for target_index, target_class in enumerate(target_cls):
        if target_index not in matched_targets and 0 <= int(target_class) < number_of_classes:
            matrix[int(target_class), number_of_classes] += 1
    for prediction_index, prediction_class in enumerate(pred_cls):
        if prediction_index not in matched_predictions and 0 <= int(prediction_class) < number_of_classes:
            matrix[number_of_classes, int(prediction_class)] += 1


def plot_precision_recall(
    true_positives: Sequence[np.ndarray],
    confidences: Sequence[np.ndarray],
    prediction_classes: Sequence[np.ndarray],
    target_classes: Sequence[np.ndarray],
    destination,
    title: str,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """Write per-class AP50 precision-recall curves."""
    tp = np.concatenate(true_positives, axis=0) if true_positives else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(confidences) if confidences else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(prediction_classes) if prediction_classes else np.zeros((0,), dtype=np.int64)
    target_cls = np.concatenate(target_classes) if target_classes else np.zeros((0,), dtype=np.int64)
    classes, counts = np.unique(target_cls, return_counts=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    for class_id, count in zip(classes, counts):
        selected = pred_cls == class_id
        if not selected.any():
            continue
        class_tp = tp[selected, 0].astype(np.float64)
        cumulative_tp = class_tp.cumsum()
        recall = cumulative_tp / max(int(count), 1)
        precision = cumulative_tp / (np.arange(class_tp.size) + 1)
        precision = np.flip(np.maximum.accumulate(np.flip(precision)))
        name = (
            str(class_names[int(class_id)])
            if class_names is not None and int(class_id) < len(class_names)
            else str(int(class_id))
        )
        axis.plot(recall, precision, label=name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title=title)
    axis.grid(True, alpha=0.25)
    if classes.size:
        axis.legend(fontsize="small")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], destination) -> None:
    """Write a row-normalized class confusion matrix with a background category."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(name) for name in class_names] + ["background"]
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sum, 1)
    figure, axis = plt.subplots(figsize=(max(7, len(names) * 0.8), max(6, len(names) * 0.7)), constrained_layout=True)
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Validation confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black", fontsize=8)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _nms_indices(boxes: torch.Tensor, scores: torch.Tensor, threshold: float, limit: int):
    """Small dependency-free NMS equivalent for one class of xyxy boxes."""
    order = scores.argsort(descending=True)
    kept = []
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp_(min=0) * (y2 - y1).clamp_(min=0)
    while order.numel() and len(kept) < limit:
        current = order[0]
        kept.append(current)
        order = order[1:]
        if not order.numel():
            break
        xx1 = torch.maximum(x1[current], x1[order])
        yy1 = torch.maximum(y1[current], y1[order])
        xx2 = torch.minimum(x2[current], x2[order])
        yy2 = torch.minimum(y2[current], y2[order])
        intersection = (xx2 - xx1).clamp_(min=0) * (yy2 - yy1).clamp_(min=0)
        overlap = intersection / (areas[current] + areas[order] - intersection + 1e-7)
        order = order[overlap <= threshold]
    return torch.stack(kept) if kept else order.new_zeros((0,), dtype=torch.long)


def pose_non_max_suppression(
    predictions: torch.Tensor,
    num_classes: int,
    num_keypoints: int,
    keypoint_dimensions: int,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    agnostic: bool = False,
):
    """Decode multi-label pose candidates and apply class-aware box NMS."""
    outputs = []
    for image_prediction in predictions.transpose(1, 2):
        boxes = image_prediction[:, :4]
        class_scores = image_prediction[:, 4:4 + num_classes]
        keypoints = image_prediction[:, 4 + num_classes:].reshape(
            -1, num_keypoints, keypoint_dimensions
        )
        anchor_indices, classes = torch.where(class_scores > confidence_threshold)
        if not anchor_indices.numel():
            outputs.append((
                boxes.new_zeros((0, 4)),
                boxes.new_zeros((0,)),
                classes,
                keypoints.new_zeros((0, num_keypoints, keypoint_dimensions)),
            ))
            continue

        boxes = boxes[anchor_indices]
        boxes_xyxy = torch.cat(
            (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1
        )
        scores = class_scores[anchor_indices, classes]
        keypoints = keypoints[anchor_indices]
        if anchor_indices.numel() > 30000:
            selected = scores.argsort(descending=True)[:30000]
            boxes_xyxy, scores, classes, keypoints = (
                boxes_xyxy[selected], scores[selected], classes[selected], keypoints[selected]
            )

        nms_classes = classes.new_zeros(classes.shape) if agnostic else classes
        kept_by_class = []
        for class_id in torch.unique(nms_classes):
            indices = torch.nonzero(nms_classes == class_id, as_tuple=False).flatten()
            kept_by_class.append(
                indices[_nms_indices(boxes_xyxy[indices], scores[indices], iou_threshold, max_detections)]
            )
        kept = (
            torch.cat(kept_by_class)
            if kept_by_class else classes.new_zeros((0,), dtype=torch.long)
        )
        kept = kept[scores[kept].argsort(descending=True)[:max_detections]]
        outputs.append((boxes_xyxy[kept], scores[kept], classes[kept], keypoints[kept]))
    return outputs


__all__ = [
    "IOU_THRESHOLDS",
    "box_iou",
    "build_coco_ground_truth",
    "evaluate_coco_keypoints",
    "keypoint_error_metrics",
    "keypoint_error_metrics_from_coco_matches",
    "keypoint_oks",
    "match_predictions",
    "pose_non_max_suppression",
    "plot_confusion_matrix",
    "plot_precision_recall",
    "summarize_ap",
    "update_confusion_matrix",
]
