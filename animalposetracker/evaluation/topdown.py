"""AnimalViTPose SimCC validation adapter."""

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from animalposetracker.evaluation.metrics import (
    OKS_BBOX_AREA_FACTOR,
    build_coco_ground_truth,
    evaluate_coco_keypoints,
    keypoint_error_metrics,
)
from animalposetracker.postprocessing.nms_core import (
    greedy_suppress,
    group_by_key,
    oks_similarity,
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
        flip_tta: bool = False,
        coco_max_detections: int = 20,
        oks_nms_threshold: float = 0.9,
        keypoint_score_threshold: float = 0.2,
        pck_threshold: float = 0.05,
        auc_norm_factor: float = 30.0,
        auc_thresholds: int = 20,
        oks_area_mode: str = "annotation",
        ap_metric: str = "COCO/AP",
        yolo_max_detections: int = 300,
    ) -> None:
        self.input_size = tuple(map(int, input_size))
        self.split_ratio = float(split_ratio)
        self.flip_indices = (
            torch.as_tensor(flip_indices, dtype=torch.long) if flip_indices is not None else None
        )
        # Flip TTA is wired into validation only. The deployed decoder
        # (postprocessing/output_decoders.decode_simcc) receives already-computed output arrays
        # and holds no reference to the model, so it cannot run the second forward pass that TTA
        # needs. Leaving TTA on here would make every reported metric an optimistic number the
        # shipped decode can never reproduce, so it is opt-in and off by default.
        self.flip_tta = bool(flip_tta)
        if self.flip_tta and self.flip_indices is None:
            warnings.warn(
                "flip_tta was requested but the dataset defines no flip_idx; "
                "validation will run without test-time augmentation",
                stacklevel=2,
            )
        self.kpt_oks_sigmas = np.asarray(kpt_oks_sigmas, dtype=np.float32)
        self.validation_dataset = validation_dataset
        self.coco_max_detections = int(coco_max_detections)
        self.oks_nms_threshold = float(oks_nms_threshold)
        self.keypoint_score_threshold = float(keypoint_score_threshold)
        self.pck_threshold = float(pck_threshold)
        self.auc_norm_factor = float(auc_norm_factor)
        self.auc_thresholds = int(auc_thresholds)
        self.oks_area_mode = str(oks_area_mode).strip().lower()
        self.ap_metric = str(ap_metric).strip()
        self.yolo_max_detections = int(yolo_max_detections)

    def __call__(self, model, loader, device):
        predictions_for_coco = []
        errors = []
        visibility_parts = []
        bbox_sizes = []
        if self.validation_dataset is None:
            raise ValueError("AnimalViTPose validation requires the validation dataset")
        coco_gt, _image_ids_by_path, _category_ids_by_class = build_coco_ground_truth(
            self.validation_dataset.annotations,
            use_coco_api=self.ap_metric == "COCO/AP",
        )

        with torch.inference_mode():
            for batch in tqdm(loader, desc="val", unit="batch", dynamic_ncols=True, leave=False):
                raw_images = batch["images"]
                images = raw_images.to(device, non_blocking=device.type == "cuda")
                # Match the trainer's uint8 -> [0, 1] scaling (engine._prepare_batch).
                if raw_images.dtype == torch.uint8:
                    images = images.float().div_(255)
                targets = batch["targets"]
                pred_x, pred_y = model(images)
                if self.flip_tta and self.flip_indices is not None:
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
                annotation_areas = targets["area"]
                if torch.is_tensor(annotation_areas):
                    annotation_areas = annotation_areas.detach().cpu().numpy()
                annotation_areas = np.asarray(annotation_areas, dtype=np.float32).reshape(-1)
                ids = targets["image_id"].numpy().tolist()
                categories = targets["category_id"].numpy().tolist()

                for sample_index in range(len(source_coords)):
                    sample_visible = visible[sample_index]
                    delta = source_coords[sample_index] - gt_coords[sample_index]
                    distance = np.linalg.norm(delta, axis=-1)
                    errors.append(distance.astype(np.float32))
                    visibility_parts.append(sample_visible)
                    bbox = boxes[sample_index]
                    box_width = max(float(bbox[2] - bbox[0]), 0.0)
                    box_height = max(float(bbox[3] - bbox[1]), 0.0)
                    bbox_sizes.append(max(box_width, box_height))
                    # OKS-NMS needs an object area, and it must sit on the same scale as the AP we
                    # report, so it follows the evaluator's own oks_area_mode switch:
                    #   "annotation" -> the GT annotation's own area, which is literally the value
                    #                   build_coco_ground_truth hands to pycocotools as gt['area'].
                    #                   The dataset already carries it in targets["area"]; its
                    #                   YOLO-label fallback equals the synthesised w*h.
                    #   "bbox"       -> OKS_BBOX_AREA_FACTOR * w * h, matching the evaluator's
                    #                   evaluator-only GT copy.
                    # This used to be prod(bbox_scale) = (1.25*max(w, h))**2 -- the padded *crop*
                    # area, which inflated the term by 1.5625x and over-suppressed boxes at 0.9.
                    detection_area = (
                        OKS_BBOX_AREA_FACTOR * box_width * box_height
                        if self.oks_area_mode == "bbox"
                        else float(annotation_areas[sample_index])
                    )

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
                            "area": detection_area,
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
            oks_area_mode=self.oks_area_mode,
            ap_metric=self.ap_metric,
            yolo_max_detections=self.yolo_max_detections,
        )

    def _oks_nms(self, detections, threshold: float):
        """Hard OKS suppression, validation-only, in the spirit of MMPose's ``oks_nms``.

        Nothing on the deployment path suppresses for AnimalViTPose --
        ``prediction.cli._predict_topdown_image`` decodes each supplied box independently -- and
        MMPose's own ``CocoMetric`` runs no NMS either (``oks_nms`` there serves multi-instance
        demo inference). A top-down crop yields exactly one detection per ground-truth instance,
        so detections are inherently distinct and this threshold rarely fires; it guards against
        duplicate/overlapping boxes handed in for the same animal.

        Each detection carries the ``area`` computed in ``__call__`` under the evaluator's
        ``oks_area_mode`` convention, so this threshold and the reported AP agree on the OKS
        scale. AnimalRTPose has no OKS-NMS at all -- it suppresses on box IoU -- so the two heads
        share no NMS area convention.

        The skeleton (group, order, greedy) and the OKS expression itself now come from
        ``postprocessing.nms_core``; only the per-detection plumbing stays here.
        """
        grouped = group_by_key([int(detection["image_id"]) for detection in detections])
        kept = []
        variances = (self.kpt_oks_sigmas * 2.0) ** 2
        for positions in grouped.values():
            instances = [detections[position] for position in positions]
            order = sorted(
                range(len(instances)),
                key=lambda index: instances[index]["score"],
                reverse=True,
            )

            def suppress(selected_index, rest_indices):
                selected = instances[selected_index]
                selected_keypoints = np.asarray(
                    selected["keypoints"], dtype=np.float32
                ).reshape(-1, 3)
                return [
                    oks_similarity(
                        selected_keypoints,
                        np.asarray(instances[index]["keypoints"], dtype=np.float32).reshape(-1, 3),
                        selected["area"],
                        instances[index]["area"],
                        variances,
                    )
                    > threshold
                    for index in rest_indices
                ]

            kept.extend(
                instances[position] for position in greedy_suppress(order, suppress=suppress)
            )
        return kept
