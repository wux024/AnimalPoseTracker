"""Ultralytics-compatible NMS for AnimalRTPose detections."""

import logging
import time

import torch

try:
    import torchvision
except (ImportError, RuntimeError):
    torchvision = None


LOGGER = logging.getLogger(__name__)


def _nms_indices(boxes: torch.Tensor, scores: torch.Tensor, threshold: float):
    """Port Ultralytics' TorchNMS.nms greedy suppression loop."""
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
            keep = _nms_indices(nms_boxes, scores, iou_threshold)
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
