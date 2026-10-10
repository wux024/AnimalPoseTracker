"""AnimalRTPose validation adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from animalposetracker.evaluation.metrics import (
    build_coco_ground_truth, evaluate_coco_keypoints,
    keypoint_error_metrics_from_coco_matches,
)
from animalposetracker.postprocessing.nms import pose_non_max_suppression


class PoseDetectionValidator:
    """Compute validation loss and standard COCO keypoint AP."""

    def __init__(
        self,
        criterion: PoseDetectionLoss,
        confidence_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        agnostic_nms: bool = False,
        plots: bool = False,
        output_dir: Optional[str] = None,
        class_names: Optional[Sequence[str]] = None,
        validation_dataset=None,
        keypoint_pck_threshold: float = 0.05,
        keypoint_auc_norm_factor: float = 30.0,
        keypoint_auc_thresholds: int = 20,
        coco_max_detections: int = 300,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("validation confidence threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("validation NMS IoU threshold must be between 0 and 1")
        if max_detections < 1:
            raise ValueError("max_detections must be at least 1")
        self.criterion = criterion
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.agnostic_nms = bool(agnostic_nms)
        self.plots = bool(plots)
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        self.class_names = list(class_names) if class_names is not None else None
        self.validation_dataset = validation_dataset
        self.keypoint_pck_threshold = float(keypoint_pck_threshold)
        self.keypoint_auc_norm_factor = float(keypoint_auc_norm_factor)
        self.keypoint_auc_thresholds = int(keypoint_auc_thresholds)
        self.coco_max_detections = int(coco_max_detections)

    def __call__(self, model, loader, device) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        batches = 0
        image_count = 0
        speed_seconds = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        coco_detections = []
        if hasattr(self.criterion, "to"):
            self.criterion.to(device)
        if self.validation_dataset is None:
            raise ValueError("COCO keypoint validation requires the validation dataset metadata")
        coco_gt, image_ids_by_path, category_ids_by_class = build_coco_ground_truth(
            self.validation_dataset
        )
        original_model_state = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for batch in loader:
                    self._synchronize(device)
                    stage_started = time.perf_counter()
                    images, targets = self._move_batch(batch, device, next(model.parameters()).dtype)
                    self._synchronize(device)
                    speed_seconds["preprocess"] += time.perf_counter() - stage_started
                    image_count += int(images.shape[0])

                    stage_started = time.perf_counter()
                    output = model(images)
                    self._synchronize(device)
                    speed_seconds["inference"] += time.perf_counter() - stage_started
                    if not isinstance(output, (tuple, list)) or len(output) != 2:
                        raise TypeError("Pose validation expects model output (decoded predictions, raw predictions)")
                    decoded_predictions, raw_predictions = output

                    stage_started = time.perf_counter()
                    result = self.criterion(raw_predictions, targets)
                    metrics = result.get("metrics", result)
                    batches += 1
                    for name, value in metrics.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            totals[name] = totals.get(name, 0.0) + float(value.float().item())
                    self._synchronize(device)
                    speed_seconds["loss"] += time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    self._collect_coco_detections(
                        decoded_predictions,
                        images,
                        targets,
                        image_ids_by_path,
                        category_ids_by_class,
                        coco_detections,
                    )
                    self._synchronize(device)
                    speed_seconds["postprocess"] += time.perf_counter() - stage_started
        finally:
            model.train(original_model_state)
        if batches == 0:
            raise ValueError("The validation data loader produced no batches")
        result = {name: value / batches for name, value in totals.items()}
        for stage, seconds in speed_seconds.items():
            result[f"speed/{stage}_ms"] = seconds * 1000.0 / max(image_count, 1)
        result["speed/inference_fps"] = image_count / max(speed_seconds["inference"], 1e-9)
        image_ids = list(image_ids_by_path.values())
        active_category_ids = (
            list(category_ids_by_class.values()) if category_ids_by_class else None
        )
        coco_metrics, matched_pairs = evaluate_coco_keypoints(
            coco_gt,
            coco_detections,
            self.criterion.keypoint_loss.kpt_oks_sigmas.detach().cpu().numpy(),
            image_ids=image_ids,
            category_ids=active_category_ids,
            use_categories=not bool(getattr(self.validation_dataset, "single_cls", False)),
            max_detections=self.coco_max_detections,
            return_matches=True,
            oks_area_mode="ultralytics_bbox",
        )
        result.update(coco_metrics)
        result.update(keypoint_error_metrics_from_coco_matches(
            matched_pairs,
            pck_threshold=self.keypoint_pck_threshold,
            auc_norm_factor=self.keypoint_auc_norm_factor,
            auc_thresholds=self.keypoint_auc_thresholds,
        ))
        return result

    @staticmethod
    def _synchronize(device) -> None:
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    def _collect_coco_detections(
        self,
        predictions: torch.Tensor,
        images: torch.Tensor,
        targets: Mapping[str, Any],
        image_ids_by_path: Mapping[str, int],
        category_ids_by_class: Mapping[int, int],
        coco_detections,
    ) -> None:
        if not torch.is_tensor(predictions) or predictions.ndim != 3:
            raise TypeError("Decoded pose predictions must be a (batch, channels, anchors) tensor")
        height, width = images.shape[-2:]
        detections = pose_non_max_suppression(
            predictions.float(),
            num_classes=self.criterion.num_classes,
            num_keypoints=self.criterion.num_keypoints,
            keypoint_dimensions=self.criterion.keypoint_dimensions,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            agnostic=self.agnostic_nms,
        )
        paths = targets.get("image_paths")
        if not isinstance(paths, (list, tuple)) or len(paths) != len(detections):
            raise ValueError("Validation batches must preserve one image path per input image")

        for image_index, (path_value, prediction) in enumerate(zip(paths, detections)):
            if isinstance(path_value, (list, tuple)):
                raise ValueError("COCO keypoint evaluation does not support augmented validation images")
            image_path = str(Path(path_value).expanduser().resolve())
            image_id = image_ids_by_path.get(image_path)
            if image_id is None:
                raise ValueError(f"No COCO image ID is mapped to validation image {image_path}")
            image_metadata = getattr(
                self.validation_dataset, "_coco_image_metadata_by_path", {}
            ).get(image_path, {})
            image_width = int(image_metadata.get("width", 0))
            image_height = int(image_metadata.get("height", 0))
            if image_width <= 0 or image_height <= 0:
                import cv2

                source = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if source is None:
                    raise ValueError(f"Could not read image dimensions for {image_path}")
                image_height, image_width = source.shape[:2]

            rect_geometry = getattr(
                self.validation_dataset, "_rect_geometry_by_path", {}
            ).get(image_path)
            if rect_geometry is not None:
                scale_x, scale_y, pad_x, pad_y = rect_geometry
            else:
                scale = min(width / image_width, height / image_height)
                resized_width = max(1, int(round(image_width * scale)))
                resized_height = max(1, int(round(image_height * scale)))
                pad_x = (width - resized_width) // 2
                pad_y = (height - resized_height) // 2
                scale_x = scale_y = scale
            pred_boxes, pred_scores, pred_classes, pred_keypoints = prediction
            for score, class_id, keypoints in zip(pred_scores, pred_classes, pred_keypoints):
                class_index = int(class_id)
                if bool(getattr(self.validation_dataset, "single_cls", False)):
                    category_id = min(category_ids_by_class.values(), default=1)
                else:
                    category_id = category_ids_by_class.get(class_index, class_index + 1)
                points = keypoints.detach().float().cpu().numpy().reshape(
                    self.criterion.num_keypoints,
                    self.criterion.keypoint_dimensions,
                )
                points_xy = points[:, :2].copy()
                points_xy[:, 0] = (points_xy[:, 0] - pad_x) / scale_x
                points_xy[:, 1] = (points_xy[:, 1] - pad_y) / scale_y
                if self.criterion.keypoint_dimensions == 3:
                    point_scores = points[:, 2]
                else:
                    point_scores = np.ones((self.criterion.num_keypoints,), dtype=np.float32)
                coco_keypoints = np.stack((points_xy[:, 0], points_xy[:, 1], point_scores), axis=-1)
                coco_detections.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "keypoints": coco_keypoints.reshape(-1).tolist(),
                    "score": float(score.detach().cpu().item()),
                })

    @staticmethod
    def _move_batch(batch, device, dtype=torch.float32):
        if not isinstance(batch, Mapping) or "images" not in batch or "targets" not in batch:
            raise TypeError("Validation batches must contain 'images' and 'targets'")
        images = batch["images"].to(
            device, dtype=dtype, non_blocking=device.type == "cuda"
        )
        targets = {
            key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
            for key, value in batch["targets"].items()
        }
        return images, targets
