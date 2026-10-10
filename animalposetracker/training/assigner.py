"""Local task-aligned target assignment for AnimalPoseTracker pose training."""

import torch
import torch.nn as nn


def aligned_complete_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return CIoU for broadcastable xyxy box tensors."""
    a = boxes_a
    b = boxes_b
    top_left = torch.maximum(a[..., :2], b[..., :2])
    bottom_right = torch.minimum(a[..., 2:], b[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    size_a = (a[..., 2:] - a[..., :2]).clamp_min(0)
    size_b = (b[..., 2:] - b[..., :2]).clamp_min(0)
    area_a = size_a.prod(dim=-1)
    area_b = size_b.prod(dim=-1)
    iou = intersection / (area_a + area_b - intersection).clamp_min(1e-9)

    center_a = (a[..., :2] + a[..., 2:]) * 0.5
    center_b = (b[..., :2] + b[..., 2:]) * 0.5
    center_distance = (center_a - center_b).square().sum(dim=-1)
    outer_min = torch.minimum(a[..., :2], b[..., :2])
    outer_max = torch.maximum(a[..., 2:], b[..., 2:])
    outer_diagonal = (outer_max - outer_min).square().sum(dim=-1).clamp_min(1e-9)
    safe_a = size_a.clamp_min(1e-7)
    safe_b = size_b.clamp_min(1e-7)
    aspect = (4.0 / (torch.pi ** 2)) * (
        torch.atan(safe_a[..., 0] / safe_a[..., 1])
        - torch.atan(safe_b[..., 0] / safe_b[..., 1])
    ).square()
    with torch.no_grad():
        alpha = aspect / (1.0 - iou + aspect).clamp_min(1e-9)
    return iou - center_distance / outer_diagonal - alpha * aspect


def pairwise_complete_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return pairwise CIoU for two xyxy box sets."""
    return aligned_complete_iou(boxes_a[:, None, :], boxes_b[None, :, :]).clamp_min(0)


class TaskAlignedAssigner(nn.Module):
    """Assign candidates using class confidence, CIoU, and in-box anchor centers."""

    def __init__(
        self,
        topk: int,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
        stride_val: float = 16.0,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stride_val = float(stride_val)
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, pred_scores, pred_boxes, anchor_centers, gt_labels, gt_boxes, gt_mask):
        batch_size, anchor_count, _ = pred_scores.shape
        max_targets = gt_boxes.shape[1]
        if max_targets == 0:
            return (
                torch.full((batch_size, anchor_count), self.num_classes, device=pred_scores.device),
                torch.zeros_like(pred_boxes),
                torch.zeros_like(pred_scores),
                torch.zeros((batch_size, anchor_count), dtype=torch.bool, device=pred_scores.device),
                torch.zeros((batch_size, anchor_count), dtype=torch.long, device=pred_scores.device),
            )

        inside = self._centers_in_boxes(anchor_centers, gt_boxes)
        align, overlaps = self._alignment_metrics(
            pred_scores, pred_boxes, gt_labels, gt_boxes, inside & gt_mask.bool()
        )
        topk = min(self.topk, anchor_count)
        topk_mask = gt_mask.expand(-1, -1, topk).bool()
        selected = self._topk_mask(align, topk_mask, topk) * inside * gt_mask
        target_gt_index, foreground, selected = self._resolve_multiple_targets(selected, overlaps)
        labels, boxes, scores = self._targets(gt_labels, gt_boxes, target_gt_index, foreground)

        selected_align = align * selected
        max_align = selected_align.amax(dim=-1, keepdim=True)
        max_overlap = (overlaps * selected).amax(dim=-1, keepdim=True)
        quality = (selected_align * max_overlap / (max_align + self.eps)).amax(dim=-2).unsqueeze(-1)
        scores = scores * quality
        return labels, boxes, scores, foreground.bool(), target_gt_index

    def _centers_in_boxes(self, anchor_centers, gt_boxes):
        center = (gt_boxes[..., :2] + gt_boxes[..., 2:]) * 0.5
        size = (gt_boxes[..., 2:] - gt_boxes[..., :2]).clamp_min(0)
        size = size.clamp_min(self.stride_val)
        left_top = (center - size * 0.5).unsqueeze(2)
        right_bottom = (center + size * 0.5).unsqueeze(2)
        deltas = torch.cat(
            (anchor_centers[None, None, :, :] - left_top, right_bottom - anchor_centers[None, None, :, :]),
            dim=-1,
        )
        return deltas.amin(dim=-1).gt(1e-9)

    def _alignment_metrics(self, pred_scores, pred_boxes, gt_labels, gt_boxes, candidate_mask):
        batch_size, max_targets, _ = gt_boxes.shape
        anchor_count = pred_boxes.shape[1]
        labels = gt_labels.squeeze(-1).long().clamp_(0, self.num_classes - 1)
        class_scores = pred_scores.transpose(1, 2)
        gather_index = labels.unsqueeze(-1).expand(-1, -1, anchor_count)
        selected_scores = class_scores.gather(1, gather_index)
        selected_scores = selected_scores * candidate_mask

        overlaps = pred_boxes.new_zeros((batch_size, max_targets, anchor_count))
        for batch_index in range(batch_size):
            overlaps[batch_index] = pairwise_complete_iou(
                gt_boxes[batch_index], pred_boxes[batch_index]
            )
        overlaps = overlaps * candidate_mask
        align = selected_scores.clamp_min(0).pow(self.alpha) * overlaps.pow(self.beta)
        return align, overlaps

    def _topk_mask(self, metrics, valid_target_mask, topk):
        values, indices = metrics.topk(topk, dim=-1, largest=True)
        topk_mask = valid_target_mask
        if topk_mask is None:
            topk_mask = (values.amax(dim=-1, keepdim=True) > self.eps).expand_as(indices)
        indices = indices.masked_fill(~topk_mask, 0)
        counts = torch.zeros_like(metrics, dtype=torch.int8)
        ones = torch.ones_like(indices[..., :1], dtype=torch.int8)
        for rank in range(topk):
            counts.scatter_add_(-1, indices[..., rank:rank + 1], ones)
        counts.masked_fill_(counts > 1, 0)
        return counts.to(metrics.dtype)

    @staticmethod
    def _resolve_multiple_targets(selected, overlaps):
        foreground = selected.sum(dim=-2)
        if foreground.max() > 1:
            multiple = (foreground.unsqueeze(1) > 1).expand_as(selected)
            best_overlap = overlaps.argmax(dim=1)
            replacement = torch.zeros_like(selected)
            replacement.scatter_(1, best_overlap.unsqueeze(1), 1)
            selected = torch.where(multiple, replacement, selected).float()
            foreground = selected.sum(dim=-2)
        target_gt_index = selected.argmax(dim=-2)
        return target_gt_index, foreground, selected

    def _targets(self, gt_labels, gt_boxes, target_gt_index, foreground):
        batch_size, anchor_count = target_gt_index.shape
        max_targets = gt_boxes.shape[1]
        batch_indices = torch.arange(batch_size, device=gt_labels.device)[:, None]
        target_labels = gt_labels.squeeze(-1).long()[batch_indices, target_gt_index]
        target_boxes = gt_boxes[batch_indices, target_gt_index]
        target_labels = target_labels.clamp_min(0)
        target_scores = torch.zeros(
            (batch_size, anchor_count, self.num_classes),
            dtype=gt_boxes.dtype,
            device=gt_boxes.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        return target_labels, target_boxes, target_scores * foreground.unsqueeze(-1)
