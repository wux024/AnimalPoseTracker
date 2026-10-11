"""Pose metrics and COCO keypoint evaluation utilities."""

import contextlib
import copy
import io
import json
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
    """Use the same 101-point AP integration as Ultralytics main."""
    return _ultralytics_compute_ap(recall, precision)

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
    """Return pycocotools classes used for standard COCO keypoint evaluation."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return COCO, COCOeval
    except ImportError as exc:
        raise ImportError(
            "COCO keypoint validation requires pycocotools; "
            "install AnimalPoseTracker with its training extra."
        ) from exc

def build_coco_ground_truth(dataset, use_coco_api: bool = True):
    """Build COCO-formatted ground-truth data and stable ID mappings.

    Native COCO annotations are kept intact so their category IDs, instance areas,
    and image IDs retain their standard meaning. YOLO pose labels are converted to
    the same in-memory schema. Construct a COCO API object only for COCO evaluation.
    """
    image_ids_by_path = dict(getattr(dataset, "image_id_by_path", {}))
    category_ids_by_class = dict(getattr(dataset, "category_id_by_class", {}))
    annotation_path = getattr(dataset, "annotation_path", None)
    if annotation_path is not None:
        if use_coco_api:
            COCO, _COCOeval = _coco_keypoint_api()
            with contextlib.redirect_stdout(io.StringIO()):
                coco_gt = COCO(str(annotation_path))
            return coco_gt, image_ids_by_path, category_ids_by_class
        with Path(annotation_path).open("r", encoding="utf-8-sig") as stream:
            return json.load(stream), image_ids_by_path, category_ids_by_class

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
    coco_dataset = {
        "info": {"description": "AnimalPoseTracker validation set"},
        "images": image_records,
        "annotations": annotations,
        "categories": categories,
    }
    if not use_coco_api:
        return coco_dataset, image_ids_by_path, category_ids_by_class
    COCO, _COCOeval = _coco_keypoint_api()
    coco_gt = COCO()
    coco_gt.dataset = coco_dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    return coco_gt, image_ids_by_path, category_ids_by_class


def _ultralytics_match_predictions(
    prediction_classes: np.ndarray,
    target_classes: np.ndarray,
    similarity: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """Match predictions and targets like Ultralytics Validator, returning 0.50 pairs."""
    correct = np.zeros((len(prediction_classes), len(thresholds)), dtype=bool)
    if not len(prediction_classes) or not len(target_classes):
        return correct, []

    similarity = np.asarray(similarity, dtype=np.float32)
    similarity *= target_classes[:, None] == prediction_classes[None, :]
    matches_at_50 = []
    for threshold_index, threshold in enumerate(thresholds):
        matches = np.argwhere(similarity >= threshold)
        if len(matches) > 1:
            matches = matches[similarity[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        if len(matches):
            correct[matches[:, 1].astype(int), threshold_index] = True
            if threshold_index == 0:
                matches_at_50 = matches
    return correct, matches_at_50


def _ultralytics_compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using Ultralytics' 101-point precision interpolation and integration."""
    modified_recall = np.concatenate(
        ([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0])
    )
    precision_envelope = np.concatenate(([1.0], precision, [0.0], [0.0]))
    precision_envelope = np.flip(np.maximum.accumulate(np.flip(precision_envelope)))
    recall_samples = np.linspace(0.0, 1.0, 101)
    sampled_precision = np.interp(recall_samples, modified_recall, precision_envelope)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(sampled_precision, recall_samples))


def _ultralytics_pose_ap(
    coco_dataset: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
    kpt_oks_sigmas: Sequence[float],
    image_ids: Optional[Sequence[int]],
    category_ids: Optional[Sequence[int]],
    use_categories: bool,
    oks_area_mode: str,
    max_detections: int,
    return_matches: bool,
):
    """Reproduce Ultralytics pose metrics directly from labels and final predictions."""
    sigma = torch.as_tensor(kpt_oks_sigmas, dtype=torch.float32).reshape(-1)
    thresholds = IOU_THRESHOLDS.cpu().numpy().astype(np.float32)
    selected_images = None if image_ids is None else {int(value) for value in image_ids}
    selected_categories = (
        {int(value) for value in category_ids}
        if use_categories and category_ids is not None else None
    )
    ground_truth_by_image: Dict[int, list] = {}
    predictions_by_image: Dict[int, list] = {}

    for annotation in coco_dataset.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if selected_images is not None and image_id not in selected_images:
            continue
        if selected_categories is not None and category_id not in selected_categories:
            continue
        ground_truth_by_image.setdefault(image_id, []).append(annotation)

    for detection in detections:
        image_id = int(detection["image_id"])
        category_id = int(detection["category_id"])
        if selected_images is not None and image_id not in selected_images:
            continue
        if selected_categories is not None and category_id not in selected_categories:
            continue
        predictions_by_image.setdefault(image_id, []).append(detection)

    all_correct = []
    all_confidence = []
    all_prediction_classes = []
    all_target_classes = []
    matched_pairs = []
    image_order = (
        [int(value) for value in image_ids]
        if image_ids is not None
        else [int(image["id"]) for image in coco_dataset.get("images", [])]
    )
    for image_id in image_order:
        ground_truth = ground_truth_by_image.get(image_id, [])
        image_predictions = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:max_detections]
        target_classes = np.asarray(
            [int(item["category_id"]) if use_categories else 0 for item in ground_truth],
            dtype=np.int64,
        )
        prediction_classes = np.asarray(
            [int(item["category_id"]) if use_categories else 0 for item in image_predictions],
            dtype=np.int64,
        )
        if ground_truth and image_predictions:
            gt_keypoints = torch.as_tensor(
                [item["keypoints"] for item in ground_truth], dtype=torch.float32
            ).reshape(len(ground_truth), -1, 3)
            pred_keypoints = torch.as_tensor(
                [item["keypoints"] for item in image_predictions], dtype=torch.float32
            ).reshape(len(image_predictions), -1, 3)
            boxes = torch.as_tensor(
                [item["bbox"] for item in ground_truth], dtype=torch.float32
            )
            if oks_area_mode == "bbox":
                area = boxes[:, 2] * boxes[:, 3] * 0.53
            else:
                area = torch.as_tensor(
                    [item["area"] for item in ground_truth], dtype=torch.float32
                )
            delta = (
                gt_keypoints[:, None, :, :2] - pred_keypoints[None, :, :, :2]
            ).square().sum(-1)
            variance = (2.0 * sigma).square()
            exponent = delta / (variance[None, None, :] * (area[:, None, None] + 1e-7) * 2.0)
            visible = gt_keypoints[..., 2] != 0
            oks = (
                torch.exp(-exponent) * visible[:, None, :]
            ).sum(-1) / (visible.sum(-1)[:, None] + 1e-7)
            correct, matches_at_50 = _ultralytics_match_predictions(
                prediction_classes,
                target_classes,
                oks.cpu().numpy(),
                thresholds,
            )
            if return_matches:
                matched_pairs.extend(
                    (ground_truth[int(target_index)], image_predictions[int(prediction_index)])
                    for target_index, prediction_index in matches_at_50
                )
        else:
            correct = np.zeros((len(image_predictions), len(thresholds)), dtype=bool)

        all_correct.append(correct)
        all_confidence.extend(float(item["score"]) for item in image_predictions)
        all_prediction_classes.extend(prediction_classes.tolist())
        all_target_classes.extend(target_classes.tolist())

    target_classes = np.asarray(all_target_classes, dtype=np.int64)
    if not target_classes.size:
        metrics = {
            "ultralytics/AP": 0.0,
            "ultralytics/AP50": 0.0,
            "ultralytics/AP75": 0.0,
            "ultralytics/P": 0.0,
            "ultralytics/R": 0.0,
        }
        return (metrics, matched_pairs) if return_matches else metrics

    correct = (
        np.concatenate(all_correct, axis=0)
        if all_correct else np.zeros((0, len(thresholds)), dtype=bool)
    )
    confidence = np.asarray(all_confidence, dtype=np.float64)
    prediction_classes = np.asarray(all_prediction_classes, dtype=np.int64)
    order = np.argsort(-confidence)
    correct = correct[order]
    confidence = confidence[order]
    prediction_classes = prediction_classes[order]

    classes, target_counts = np.unique(target_classes, return_counts=True)
    class_ap = np.zeros((len(classes), len(thresholds)), dtype=np.float64)
    confidence_axis = np.linspace(0.0, 1.0, 1000)
    precision_curves = np.zeros((len(classes), confidence_axis.size), dtype=np.float64)
    recall_curves = np.zeros_like(precision_curves)
    for class_index, class_id in enumerate(classes):
        class_predictions = prediction_classes == class_id
        if not np.any(class_predictions):
            continue
        class_correct = correct[class_predictions]
        false_positives = (1 - class_correct).cumsum(0)
        true_positives = class_correct.cumsum(0)
        recall = true_positives / (int(target_counts[class_index]) + 1e-16)
        precision = true_positives / (true_positives + false_positives)
        recall_curves[class_index] = np.interp(
            -confidence_axis, -confidence[class_predictions], recall[:, 0], left=0.0
        )
        precision_curves[class_index] = np.interp(
            -confidence_axis, -confidence[class_predictions], precision[:, 0], left=1.0
        )
        for threshold_index in range(len(thresholds)):
            class_ap[class_index, threshold_index] = _ultralytics_compute_ap(
                recall[:, threshold_index], precision[:, threshold_index]
            )

    f1_curve = 2.0 * precision_curves * recall_curves / (
        precision_curves + recall_curves + 1e-16
    )
    best_confidence_index = int(_smooth(f1_curve.mean(axis=0), fraction=0.1).argmax())
    metrics = {
        "ultralytics/AP": float(class_ap.mean()),
        "ultralytics/AP50": float(class_ap[:, 0].mean()),
        "ultralytics/AP75": float(class_ap[:, 5].mean()),
        "ultralytics/P": float(precision_curves[:, best_confidence_index].mean()),
        "ultralytics/R": float(recall_curves[:, best_confidence_index].mean()),
    }
    return (metrics, matched_pairs) if return_matches else metrics


def evaluate_coco_keypoints(
    coco_gt,
    detections: Sequence[Dict[str, Any]],
    kpt_oks_sigmas: Sequence[float],
    image_ids: Optional[Sequence[int]] = None,
    category_ids: Optional[Sequence[int]] = None,
    use_categories: bool = True,
    max_detections: int = 20,
    return_matches: bool = False,
    oks_area_mode: str = "annotation",
    ap_metric: str = "coco/AP",
    ultralytics_max_detections: int = 300,
):
    """Evaluate keypoint predictions with the selected COCO or Ultralytics AP path.

    When requested, also return GT/detection pairs matched at OKS 0.50 for PCK/AUC/EPE.
    ``annotation`` uses the annotated GT area. ``bbox`` uses 0.53 times the GT
    bounding-box area. Ultralytics mode uses its local matcher and does not instantiate
    COCOeval; COCO mode delegates AP and matching to pycocotools.
    """
    if oks_area_mode not in {"annotation", "bbox"}:
        raise ValueError("oks_area_mode must be 'annotation' or 'bbox'")
    if ap_metric not in {"coco/AP", "ultralytics/AP"}:
        raise ValueError("ap_metric must be 'coco/AP' or 'ultralytics/AP'")
    if ap_metric == "coco/AP" and int(max_detections) < 1:
        raise ValueError("max_detections must be at least 1")
    if ap_metric == "ultralytics/AP" and int(ultralytics_max_detections) < 1:
        raise ValueError("ultralytics_max_detections must be at least 1")
    sigma_values = np.asarray(kpt_oks_sigmas, dtype=np.float32).reshape(-1)
    if not sigma_values.size or not np.isfinite(sigma_values).all() or np.any(sigma_values <= 0):
        raise ValueError("Keypoint AP evaluation needs one finite positive OKS sigma per keypoint")
    if ap_metric == "ultralytics/AP":
        coco_dataset = coco_gt.dataset if hasattr(coco_gt, "dataset") else coco_gt
        metrics, matched_pairs = _ultralytics_pose_ap(
            coco_dataset,
            detections,
            sigma_values,
            image_ids,
            category_ids,
            use_categories,
            oks_area_mode,
            int(ultralytics_max_detections),
            return_matches,
        )
        return (metrics, matched_pairs) if return_matches else metrics

    COCO, COCOeval = _coco_keypoint_api()
    with contextlib.redirect_stdout(io.StringIO()):
        eval_coco_gt = coco_gt
        if oks_area_mode == "bbox":
            # pycocotools has no use_area option, so prepare an evaluator-only GT copy.
            dataset = copy.deepcopy(coco_gt.dataset)
            for annotation in dataset.get("annotations", []):
                bbox = np.asarray(annotation.get("bbox", ()), dtype=np.float32).reshape(-1)
                if bbox.shape != (4,) or not np.isfinite(bbox).all():
                    raise ValueError(
                        "BBox-area OKS evaluation requires finite COCO xywh bboxes"
                    )
                width = max(np.float32(bbox[2]), np.float32(0.0))
                height = max(np.float32(bbox[3]), np.float32(0.0))
                annotation["area"] = float(np.float32(0.53) * width * height)
            eval_coco_gt = COCO()
            eval_coco_gt.dataset = dataset
            eval_coco_gt.createIndex()

        if detections:
            coco_dt = eval_coco_gt.loadRes(list(detections))
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "info": eval_coco_gt.dataset.get("info", {}),
                "images": list(eval_coco_gt.dataset.get("images", [])),
                "categories": list(eval_coco_gt.dataset.get("categories", [])),
                "annotations": [],
            }
            coco_dt.createIndex()

        evaluator = COCOeval(eval_coco_gt, coco_dt, "keypoints")
        evaluator.params.kpt_oks_sigmas = sigma_values
        evaluator.params.iouThrs = IOU_THRESHOLDS.cpu().numpy().astype(np.float64)
        # Keep the COCO keypoint default (20) so stock summarize() remains valid,
        # and accumulate the requested cap as an additional operating point.
        evaluator.params.maxDets = sorted({20, int(max_detections)})
        if image_ids is not None:
            evaluator.params.imgIds = sorted({int(image_id) for image_id in image_ids})
        if category_ids is not None:
            evaluator.params.catIds = sorted({int(category_id) for category_id in category_ids})
        evaluator.params.useCats = int(bool(use_categories))
        evaluator.evaluate()
        evaluator.accumulate()
        recall = np.asarray(evaluator.eval["recall"], dtype=np.float64)
        area_index = evaluator.params.areaRngLbl.index("all")
        max_det_index = evaluator.params.maxDets.index(int(max_detections))
        iou_thresholds = np.asarray(evaluator.params.iouThrs, dtype=np.float64)

        def _mean_valid(values):
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else 0.0

        recall = recall[:, :, area_index, max_det_index]
        threshold_50 = int(np.argmin(np.abs(iou_thresholds - 0.50)))
        threshold_75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))
        metrics = {"coco/AR": _mean_valid(recall)}
        if ap_metric == "coco/AP":
            precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
            precision = precision[:, :, :, area_index, max_det_index]
            metrics.update({
                "coco/AP": _mean_valid(precision),
                "coco/AP50": _mean_valid(precision[threshold_50]),
                "coco/AP75": _mean_valid(precision[threshold_75]),
            })
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
            unmatched = int(ground_truth_id) == 0
            if unmatched or bool(is_ignored):
                continue
            matched_pairs.append((
                eval_coco_gt.anns[int(ground_truth_id)],
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
    """Compute PCK/AUC/EPE over evaluator-matched instances at OKS 0.50."""
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

__all__ = [
    "IOU_THRESHOLDS", "box_iou", "build_coco_ground_truth",
    "evaluate_coco_keypoints", "keypoint_error_metrics",
    "keypoint_error_metrics_from_coco_matches", "keypoint_oks",
    "match_predictions", "plot_confusion_matrix", "plot_precision_recall",
    "summarize_ap", "update_confusion_matrix",
]
