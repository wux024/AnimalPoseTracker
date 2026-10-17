"""One-stage (RTMO) validation adapter.

Metrics only: the head's training-time outputs (the keypoint-offset branch that
feeds positive-sample selection) only exist while the model is in training mode,
which is why the reference evaluation -- and this adapter -- reports AP / PCK /
AUC / EPE and no validation loss (the SimCC validator works the same way).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from animalposetracker.evaluation.metrics import (
    build_coco_ground_truth,
    evaluate_coco_keypoints,
    keypoint_error_metrics_from_coco_matches,
)


class RTMOPoseValidator:
    """Decode one-stage bottom-up outputs and score standard COCO keypoint AP."""

    def __init__(
        self,
        criterion,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.65,
        nms_pre: int = 100000,
        class_names: Optional[Sequence[str]] = None,
        validation_dataset=None,
        keypoint_pck_threshold: float = 0.05,
        keypoint_auc_norm_factor: float = 30.0,
        keypoint_auc_thresholds: int = 20,
        coco_max_detections: int = 20,
        oks_area_mode: str = "annotation",
        ap_metric: str = "COCO/AP",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("validation score threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("validation NMS IoU threshold must be between 0 and 1")
        if nms_pre < 1:
            raise ValueError("nms_pre must be at least 1")
        self.criterion = criterion
        self.head = criterion.head
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.nms_pre = int(nms_pre)
        self.class_names = list(class_names) if class_names is not None else None
        self.validation_dataset = validation_dataset
        self.keypoint_pck_threshold = float(keypoint_pck_threshold)
        self.keypoint_auc_norm_factor = float(keypoint_auc_norm_factor)
        self.keypoint_auc_thresholds = int(keypoint_auc_thresholds)
        self.coco_max_detections = int(coco_max_detections)
        self.oks_area_mode = str(oks_area_mode).strip().lower()
        self.ap_metric = str(ap_metric).strip()

    def __call__(self, model, loader, device) -> Dict[str, float]:
        if self.validation_dataset is None:
            raise ValueError("COCO keypoint validation requires the validation dataset metadata")
        coco_gt, image_ids_by_path, category_ids_by_class = build_coco_ground_truth(
            self.validation_dataset,
            use_coco_api=self.ap_metric == "COCO/AP",
        )
        coco_dataset = coco_gt.dataset if hasattr(coco_gt, "dataset") else coco_gt
        coco_detections = []
        image_count = 0
        speed_seconds = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

        original_model_state = model.training
        # Decode with the head of the model being evaluated: validation may run the EMA
        # copy, and its head is a different object (the engine only switched the evaluated
        # copy to eval mode, the training model stays in train mode with its BatchNorm1d
        # updating -- which aborts as soon as a single instance is decoded).
        head = model.model[-1] if getattr(model, "model", None) is not None else self.head
        model.eval()
        try:
            with torch.inference_mode():
                for batch in tqdm(loader, desc="val", unit="batch", dynamic_ncols=True, leave=False):
                    self._synchronize(device)
                    stage_started = time.perf_counter()
                    images, samples = self._move_batch(batch, device, next(model.parameters()).dtype)
                    self._synchronize(device)
                    speed_seconds["preprocess"] += time.perf_counter() - stage_started
                    image_count += int(images.shape[0])

                    stage_started = time.perf_counter()
                    outputs = model(images)
                    self._synchronize(device)
                    speed_seconds["inference"] += time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    results = head.predict_from_outputs(
                        outputs,
                        samples,
                        test_cfg=dict(
                            score_thr=self.confidence_threshold,
                            nms_thr=self.iou_threshold,
                            nms_pre=self.nms_pre,
                        ),
                    )
                    self._collect_coco_detections(
                        results,
                        images,
                        samples,
                        image_ids_by_path,
                        category_ids_by_class,
                        coco_detections,
                    )
                    self._synchronize(device)
                    speed_seconds["postprocess"] += time.perf_counter() - stage_started
        finally:
            model.train(original_model_state)

        result: Dict[str, float] = {}
        result["images"] = image_count
        result["instances"] = len(coco_dataset.get("annotations", []))
        for stage, seconds in speed_seconds.items():
            result[f"speed/{stage}_ms"] = seconds * 1000.0 / max(image_count, 1)
        image_ids = list(image_ids_by_path.values())
        active_category_ids = (
            list(category_ids_by_class.values()) if category_ids_by_class else None
        )
        coco_metrics, matched_pairs = evaluate_coco_keypoints(
            coco_gt,
            coco_detections,
            self.criterion.kpt_oks_sigmas.detach().cpu().numpy(),
            image_ids=image_ids,
            category_ids=active_category_ids,
            use_categories=not bool(getattr(self.validation_dataset, "single_cls", False)),
            max_detections=self.coco_max_detections,
            return_matches=True,
            oks_area_mode=self.oks_area_mode,
            ap_metric=self.ap_metric,
            yolo_max_detections=self.coco_max_detections,
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
        results: Sequence[Mapping[str, Any]],
        images: torch.Tensor,
        batch_samples: Sequence[Mapping[str, Any]],
        image_ids_by_path: Mapping[str, int],
        category_ids_by_class: Mapping[int, int],
        coco_detections,
    ) -> None:
        """Map canvas-space predictions back to original image pixels.

        Same geometry contract as the AnimalRTPose validator: the validation
        dataset records its rect/letterbox transform per image, with a centred
        letterbox fallback for projects that do not use rect batches.
        """
        if len(results) != len(batch_samples):
            raise ValueError("The one-stage head must return one result per input image")
        height, width = images.shape[-2:]
        for sample, prediction in zip(batch_samples, results):
            metainfo = sample.get("metainfo", {})
            path_value = metainfo.get("image_path")
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

            keypoints = prediction["keypoints"].detach().float().cpu().numpy()
            visible = prediction["keypoints_visible"].detach().float().cpu().numpy()
            for score, class_id, points, point_visibility in zip(
                prediction["scores"], prediction["labels"], keypoints, visible
            ):
                class_index = int(class_id)
                if bool(getattr(self.validation_dataset, "single_cls", False)):
                    category_id = min(category_ids_by_class.values(), default=1)
                else:
                    category_id = category_ids_by_class.get(class_index, class_index + 1)
                points_xy = points[:, :2].copy()
                points_xy[:, 0] = (points_xy[:, 0] - pad_x) / scale_x
                points_xy[:, 1] = (points_xy[:, 1] - pad_y) / scale_y
                coco_keypoints = np.stack(
                    (points_xy[:, 0], points_xy[:, 1], np.clip(point_visibility, 0.0, 1.0)),
                    axis=-1,
                )
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
        # Same pixel contract as the training loop: uint8 [0, 255] in, [0, 1] to the model.
        if batch["images"].dtype == torch.uint8:
            images = images.div_(255)
        samples = []
        for sample in batch["targets"]:
            labels = sample.get("gt_instance_labels", {})
            moved = {
                "gt_instance_labels": {
                    key: value.to(device, non_blocking=device.type == "cuda")
                    if torch.is_tensor(value) else value
                    for key, value in labels.items()
                },
                "metainfo": dict(sample.get("metainfo", {})),
            }
            samples.append(moved)
        return images, samples
