"""Pose candidate non-maximum suppression."""

import torch


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

__all__ = ["pose_non_max_suppression"]
