"""Ultralytics-compatible NMS for AnimalRTPose detections."""

import logging
import time

import torch

from animalposetracker.postprocessing.nms_core import nms_indices_torch
from animalposetracker.postprocessing.nms_params import (
    TRAINING_NMS_AGNOSTIC,
    TRAINING_NMS_CONF,
    TRAINING_NMS_IOU,
    TRAINING_NMS_MAX_DET,
)

try:
    import torchvision
except (ImportError, RuntimeError):
    torchvision = None


LOGGER = logging.getLogger(__name__)


def pose_non_max_suppression(
    predictions: torch.Tensor,
    num_classes: int,
    num_keypoints: int,
    keypoint_dimensions: int,
    confidence_threshold: float = TRAINING_NMS_CONF,
    iou_threshold: float = TRAINING_NMS_IOU,
    max_detections: int = TRAINING_NMS_MAX_DET,
    agnostic: bool = TRAINING_NMS_AGNOSTIC,
):
    """Apply Ultralytics PoseValidator's multi-label box NMS to pose predictions."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")
    if predictions.ndim != 3:
        raise ValueError("pose predictions must have shape (batch, channels, candidates)")

    batch_size = predictions.shape[0]
    class_end = 4 + int(num_classes)
    extra_size = predictions.shape[1] - class_end
    candidates = predictions[:, 4:class_end].amax(1) > confidence_threshold
    multi_label = num_classes > 1
    max_candidates = 30000
    class_offset = 7680
    time_limit = 2.0 + 0.05 * batch_size
    start_time = time.time()
    use_torchvision = predictions.device.type != "xpu" and torchvision is not None
    outputs = [(
        predictions.new_zeros((0, 4)),
        predictions.new_zeros((0,)),
        torch.zeros((0,), dtype=torch.long, device=predictions.device),
        predictions.new_zeros((0, num_keypoints, keypoint_dimensions)),
    ) for _ in range(batch_size)]

    for image_index, image_prediction in enumerate(predictions.transpose(-1, -2)):
        image_prediction = image_prediction[candidates[image_index]]
        if not image_prediction.shape[0]:
            continue

        xywh = image_prediction[:, :4].clone()
        image_prediction[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
        image_prediction[:, 2:4] = xywh[:, :2] + xywh[:, 2:] / 2
        boxes, class_scores, extra = image_prediction.split(
            (4, num_classes, extra_size), dim=1
        )

        if multi_label:
            anchor_indices, class_indices = torch.where(class_scores > confidence_threshold)
            image_prediction = torch.cat((
                boxes[anchor_indices],
                class_scores[anchor_indices, class_indices, None],
                class_indices[:, None].to(boxes.dtype),
                extra[anchor_indices],
            ), dim=1)
        else:
            scores, class_indices = class_scores.max(1, keepdim=True)
            keep = scores.view(-1) > confidence_threshold
            image_prediction = torch.cat((boxes, scores, class_indices.to(boxes.dtype), extra), dim=1)[keep]

        if image_prediction.shape[0] > max_candidates:
            order = image_prediction[:, 4].argsort(descending=True)[:max_candidates]
            image_prediction = image_prediction[order]

        class_ids = image_prediction[:, 5:6] * (0 if agnostic else class_offset)
        nms_boxes = image_prediction[:, :4] + class_ids
        scores = image_prediction[:, 4]
        if use_torchvision:
            keep = torchvision.ops.nms(nms_boxes, scores, iou_threshold)
        else:
            keep = nms_indices_torch(nms_boxes, scores, iou_threshold)
        image_prediction = image_prediction[keep[:max_detections]]

        keypoints = image_prediction[:, 6:].reshape(
            -1, num_keypoints, keypoint_dimensions
        )
        outputs[image_index] = (
            image_prediction[:, :4],
            image_prediction[:, 4],
            image_prediction[:, 5].long(),
            keypoints,
        )
        if time.time() - start_time > time_limit:
            LOGGER.warning("NMS time limit %.3fs exceeded", time_limit)
            break

    return outputs


__all__ = ["pose_non_max_suppression"]
