"""AnimalViTPose SimCC validation adapter."""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from animalposetracker.evaluation.metrics import (
    build_coco_ground_truth,
    evaluate_coco_keypoints,
    keypoint_error_metrics,
)

class SimCCPoseValidator:
    """Compute MMPose-compatible SimCC decoding and keypoint metrics."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        split_ratio: float,
        flip_indices: Optional[Sequence[int]],
        kpt_oks_sigmas: Sequence[float],
        validation_dataset,
        coco_max_detections: int = 20,
        oks_nms_threshold: float = 0.9,
        keypoint_score_threshold: float = 0.2,
        pck_threshold: float = 0.05,
        auc_norm_factor: float = 30.0,
        auc_thresholds: int = 20,
    ) -> None:
        self.input_size = tuple(map(int, input_size))
        self.split_ratio = float(split_ratio)
        self.flip_indices = (
            torch.as_tensor(flip_indices, dtype=torch.long) if flip_indices is not None else None
        )
        self.kpt_oks_sigmas = np.asarray(kpt_oks_sigmas, dtype=np.float32)
        self.validation_dataset = validation_dataset
        self.coco_max_detections = int(coco_max_detections)
        self.oks_nms_threshold = float(oks_nms_threshold)
        self.keypoint_score_threshold = float(keypoint_score_threshold)
        self.pck_threshold = float(pck_threshold)
        self.auc_norm_factor = float(auc_norm_factor)
        self.auc_thresholds = int(auc_thresholds)

    def __call__(self, model, loader, device):
        predictions_for_coco = []
        errors = []
        visibility_parts = []
        bbox_sizes = []
        if self.validation_dataset is None:
            raise ValueError("AnimalViTPose validation requires the validation dataset")
        coco_gt, _image_ids_by_path, _category_ids_by_class = build_coco_ground_truth(
            self.validation_dataset.annotations
        )

        with torch.inference_mode():
            for batch in loader:
                images = batch["images"].to(device, non_blocking=device.type == "cuda")
                targets = batch["targets"]
                pred_x, pred_y = model(images)
                if self.flip_indices is not None:
                    flipped_x, flipped_y = model(torch.flip(images, dims=(-1,)))
                    permutation = self.flip_indices.to(pred_x.device)
                    flipped_x = flipped_x.flip(dims=(-1,)).index_select(1, permutation)
                    flipped_y = flipped_y.index_select(1, permutation)
                    pred_x = (pred_x + flipped_x) * 0.5
                    pred_y = (pred_y + flipped_y) * 0.5

                max_x, coords_x = pred_x.max(dim=-1)
                max_y, coords_y = pred_y.max(dim=-1)
                keypoint_scores = torch.minimum(max_x, max_y)
                crop_coords = torch.stack((coords_x, coords_y), dim=-1).float()
                crop_coords /= self.split_ratio
                invalid = keypoint_scores <= 0
                crop_coords[invalid] = -1.0 / self.split_ratio
                crop_coords = crop_coords.detach().cpu().numpy()
                keypoint_scores = keypoint_scores.detach().cpu().numpy()
                inverse = targets["warp_inverse"].numpy()
                homogeneous = np.concatenate(
                    [crop_coords, np.ones((*crop_coords.shape[:2], 1), dtype=np.float32)], axis=-1
                )
                source_coords = np.einsum("bij,bkj->bki", inverse, homogeneous)

                gt_coords = targets["keypoints"].numpy()
                visible = targets["keypoints_visible"].numpy() > 0
                boxes = targets["bbox_xyxy"].numpy()
                bbox_scales = targets["bbox_scale"].numpy()
                ids = targets["image_id"].numpy().tolist()
                categories = targets["category_id"].numpy().tolist()

                for sample_index in range(len(source_coords)):
                    sample_visible = visible[sample_index]
                    delta = source_coords[sample_index] - gt_coords[sample_index]
                    distance = np.linalg.norm(delta, axis=-1)
                    errors.append(distance.astype(np.float32))
                    visibility_parts.append(sample_visible)
                    bbox = boxes[sample_index]
                    bbox_sizes.append(max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])))

                    if coco_gt is not None:
                        sample_scores = keypoint_scores[sample_index]
                        keypoint_array = np.concatenate(
                            [source_coords[sample_index], sample_scores[:, None]], axis=1
                        )
                        valid_scores = sample_scores[
                            sample_scores > self.keypoint_score_threshold
                        ]
                        confidence = float(valid_scores.mean()) if valid_scores.size else 0.0
                        predictions_for_coco.append({
                            "image_id": int(ids[sample_index]),
                            "category_id": int(categories[sample_index]),
                            "keypoints": keypoint_array.reshape(-1).tolist(),
                            "score": confidence,
                            "area": float(np.prod(bbox_scales[sample_index])),
                        })

        if not errors:
            raise ValueError("Validation data loader produced no top-down samples")
        errors = np.stack(errors)
        visible = np.stack(visibility_parts)
        bbox_sizes = np.maximum(np.asarray(bbox_sizes, dtype=np.float32), 1.0)
        metrics = keypoint_error_metrics(
            errors,
            visible,
            bbox_sizes,
            pck_threshold=self.pck_threshold,
            auc_norm_factor=self.auc_norm_factor,
            auc_thresholds=self.auc_thresholds,
        )
        metrics.update(self._coco_metrics(coco_gt, predictions_for_coco))
        return metrics

    def _coco_metrics(self, coco_gt, detections):
        detections = self._oks_nms(detections, threshold=self.oks_nms_threshold)
        return evaluate_coco_keypoints(
            coco_gt,
            detections,
            self.kpt_oks_sigmas,
            max_detections=self.coco_max_detections,
        )

    def _oks_nms(self, detections, threshold: float):
        """Match MMPose CocoMetric's default hard OKS NMS for top-down predictions."""
        grouped = {}
        for detection in detections:
            group_key = int(detection["image_id"])
            grouped.setdefault(group_key, []).append(detection)
        kept = []
        variances = (self.kpt_oks_sigmas * 2.0) ** 2
        for instances in grouped.values():
            order = sorted(
                range(len(instances)),
                key=lambda index: instances[index]["score"],
                reverse=True,
            )
            image_kept = []
            while order:
                selected_index = order[0]
                selected = instances[selected_index]
                image_kept.append(selected)
                selected_keypoints = np.asarray(selected["keypoints"], dtype=np.float32).reshape(-1, 3)
                remaining = []
                for candidate_index in order[1:]:
                    candidate = instances[candidate_index]
                    candidate_keypoints = np.asarray(candidate["keypoints"], dtype=np.float32).reshape(-1, 3)
                    delta = candidate_keypoints[:, :2] - selected_keypoints[:, :2]
                    denominator = (
                        variances
                        * ((float(selected["area"]) + float(candidate["area"])) * 0.5 + np.spacing(1))
                        * 2.0
                    )
                    oks = float(np.exp(-np.sum(delta * delta, axis=1) / denominator).mean())
                    if oks <= threshold:
                        remaining.append(candidate_index)
                order = remaining
            kept.extend(image_kept)
        return kept
