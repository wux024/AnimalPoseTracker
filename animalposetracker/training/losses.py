"""Pose losses following the project's pinned AnimalRTPose training behavior."""

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from animalposetracker.nn.tal import dist2bbox, make_anchors
from .assigner import COCO_KEYPOINT_SIGMAS, TaskAlignedAssigner, aligned_complete_iou, bbox_overlaps


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half = boxes[..., 2:4] * 0.5
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)

def _boxes_to_distribution_targets(
    anchor_points: torch.Tensor,
    boxes: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    left_top = anchor_points - boxes[..., :2]
    right_bottom = boxes[..., 2:] - anchor_points
    return torch.cat((left_top, right_bottom), dim=-1).clamp_(0, reg_max - 1 - 0.01)

class DistributionFocalLoss(nn.Module):
    """Interpolated cross-entropy for left/top/right/bottom distance bins."""

    def __init__(self, reg_max: int) -> None:
        super().__init__()
        self.reg_max = int(reg_max)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        lower = target.long()
        upper = lower + 1
        lower_weight = upper.to(target.dtype) - target
        upper_weight = 1.0 - lower_weight
        log_probabilities = F.log_softmax(logits, dim=1)
        lower_loss = log_probabilities.gather(1, lower.reshape(-1, 1)).reshape_as(target)
        upper_loss = log_probabilities.gather(1, upper.reshape(-1, 1)).reshape_as(target)
        return -(
            lower_loss * lower_weight + upper_loss * upper_weight
        ).mean(-1, keepdim=True)

class KeypointOKSLoss(nn.Module):
    """OKS-shaped keypoint localization loss and per-instance visibility weighting."""

    def __init__(self, kpt_oks_sigmas: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer(
            "kpt_oks_sigmas",
            torch.as_tensor(kpt_oks_sigmas, dtype=torch.float32).flatten(),
        )

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        visible_mask: torch.Tensor,
        instance_area: torch.Tensor,
    ) -> torch.Tensor:
        squared_distance = (
            (predicted[..., 0] - target[..., 0]).square()
            + (predicted[..., 1] - target[..., 1]).square()
        )
        visible_count = visible_mask.sum(dim=1)
        keypoint_factor = visible_mask.shape[1] / (visible_count + 1e-9)
        sigma_term = (2.0 * self.kpt_oks_sigmas).square().view(1, -1)
        error = squared_distance / (sigma_term * (instance_area + 1e-9) * 2.0)
        return (
            keypoint_factor.view(-1, 1)
            * ((1.0 - torch.exp(-error)) * visible_mask.to(error.dtype))
        ).mean()

class PoseDetectionLoss(nn.Module):
    """Local multi-scale pose loss aligned with the custom AnimalRTPose branch.

    It uses task-aligned top-k assignment, class BCE, CIoU, DFL, OKS-shaped
    keypoint localization loss, and a visibility BCE term. The pose-specific
    gains come from the same saved training configuration as the GUI.
    """

    loss_is_batch_sum = True

    def __init__(
        self,
        head,
        image_size: int,
        kpt_oks_sigmas: Optional[Sequence[float]] = None,
        top_k: int = 10,
        box_weight: float = 7.5,
        class_weight: float = 0.5,
        distribution_weight: float = 1.5,
        keypoint_weight: float = 12.0,
        visibility_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if getattr(head, "rle", False):
            raise NotImplementedError("RLE pose heads need their separate likelihood loss")
        self.num_classes = int(head.nc)
        self.num_keypoints, self.keypoint_dimensions = map(int, head.kpt_shape)
        self.reg_max = int(head.reg_max)
        self.image_size = int(image_size)
        self.top_k = int(top_k)
        if self.image_size < 1 or self.top_k < 1:
            raise ValueError("image_size and top_k must be positive")
        if kpt_oks_sigmas is None:
            kpt_oks_sigmas = [1.0 / self.num_keypoints] * self.num_keypoints
        if len(kpt_oks_sigmas) != self.num_keypoints:
            raise ValueError(
                f"Expected {self.num_keypoints} OKS sigmas, got {len(kpt_oks_sigmas)}"
            )

        self.box_weight = float(box_weight)
        self.class_weight = float(class_weight)
        self.distribution_weight = float(distribution_weight)
        self.keypoint_weight = float(keypoint_weight)
        self.visibility_weight = float(visibility_weight)
        self.register_buffer("strides", torch.as_tensor(head.stride, dtype=torch.float32).detach().clone())
        self.register_buffer("projection", torch.arange(self.reg_max, dtype=torch.float32))
        stride_values = torch.as_tensor(head.stride, dtype=torch.float32).flatten()
        stride_val = float(stride_values[1 if stride_values.numel() > 1 else 0])
        self.assigner = TaskAlignedAssigner(
            self.top_k, self.num_classes, alpha=0.5, beta=6.0, stride_val=stride_val
        )
        self.distribution_loss = DistributionFocalLoss(self.reg_max)
        self.keypoint_loss = KeypointOKSLoss(kpt_oks_sigmas)
        self.visibility_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(predictions, (tuple, list)) or len(predictions) != 2:
            raise TypeError("PoseDetectionLoss expects (multi_scale_predictions, keypoint_predictions)")
        feature_predictions, keypoint_predictions = predictions
        if not isinstance(feature_predictions, (tuple, list)) or not feature_predictions:
            raise TypeError("The pose head must return a non-empty list of feature predictions")

        batch_size = feature_predictions[0].shape[0]
        flattened = torch.cat(
            [feature.permute(0, 2, 3, 1).reshape(batch_size, -1, feature.shape[1])
             for feature in feature_predictions],
            dim=1,
        )
        expected_channels = self.reg_max * 4 + self.num_classes
        if flattened.shape[-1] != expected_channels:
            raise ValueError(
                f"Pose head emitted {flattened.shape[-1]} channels; expected {expected_channels}"
            )
        if keypoint_predictions.shape[-1] != flattened.shape[1]:
            raise ValueError("Keypoint and box heads emitted different anchor counts")

        pred_distribution, pred_classes = flattened.split(
            (self.reg_max * 4, self.num_classes), dim=-1
        )
        pred_distribution = pred_distribution.reshape(batch_size, -1, 4, self.reg_max)
        pred_keypoints = keypoint_predictions.transpose(1, 2).contiguous().reshape(
            batch_size, -1, self.num_keypoints, self.keypoint_dimensions
        )
        anchor_points, stride_per_anchor = make_anchors(
            feature_predictions, self.strides.to(feature_predictions[0].device), 0.5
        )
        first_stride = float(self.strides[0].item())
        input_height = int(feature_predictions[0].shape[-2] * first_stride)
        input_width = int(feature_predictions[0].shape[-1] * first_stride)

        distances = (
            pred_distribution.softmax(dim=-1)
            * self.projection.to(
                device=pred_distribution.device, dtype=pred_distribution.dtype
            )
        ).sum(dim=-1)
        pred_boxes_grid = dist2bbox(
            distances, anchor_points.unsqueeze(0), xywh=False, dim=-1
        )
        pred_boxes_pixels = pred_boxes_grid * stride_per_anchor.unsqueeze(0)
        gt_labels, gt_boxes, gt_keypoints, gt_mask = self._prepare_targets(
            targets, batch_size, input_height, input_width
        )

        target_labels, target_boxes, target_scores, foreground, target_gt_index = self.assigner(
            pred_classes.detach().sigmoid(),
            pred_boxes_pixels.detach().to(gt_boxes.dtype),
            anchor_points * stride_per_anchor,
            gt_labels,
            gt_boxes,
            gt_mask,
        )
        del target_labels
        target_score_sum = target_scores.sum().clamp_min(1.0)
        class_loss = F.binary_cross_entropy_with_logits(
            pred_classes.float(), target_scores.float(), reduction="sum"
        ) / target_score_sum.float()

        # Autocast may produce FP16 predictions. Reduce in FP32 before scaling by
        # zero so a no-foreground batch cannot overflow its FP16 sum into NaN.
        zero = pred_distribution.sum(dtype=torch.float32) * 0.0
        box_loss = zero
        dfl_loss = zero
        pose_loss = zero
        keypoint_object_loss = zero
        if foreground.any():
            target_boxes_grid = target_boxes / stride_per_anchor.unsqueeze(0)
            positive_scores = target_scores.sum(dim=-1)[foreground].unsqueeze(-1)
            iou = aligned_complete_iou(
                pred_boxes_grid[foreground], target_boxes_grid[foreground]
            ).unsqueeze(-1)
            box_loss = ((1.0 - iou) * positive_scores).sum() / target_score_sum

            target_distances = _boxes_to_distances(
                anchor_points.unsqueeze(0), target_boxes_grid, self.reg_max - 1
            )
            distribution_logits = pred_distribution[foreground].reshape(-1, self.reg_max)
            dfl_items = self.distribution_loss(
                distribution_logits, target_distances[foreground]
            )
            dfl_loss = (dfl_items * positive_scores).sum() / target_score_sum

            gt_indices = target_gt_index
            gather_index = gt_indices[..., None, None].expand(
                -1, -1, self.num_keypoints, self.keypoint_dimensions
            )
            selected_keypoints = gt_keypoints.gather(1, gather_index)
            selected_keypoints = selected_keypoints.clone()
            selected_keypoints[..., :2] /= stride_per_anchor.view(1, -1, 1, 1)
            pred_keypoints_grid = self._decode_keypoints(pred_keypoints, anchor_points)
            positive_gt_keypoints = selected_keypoints[foreground]
            positive_pred_keypoints = pred_keypoints_grid[foreground]
            if self.keypoint_dimensions == 3:
                keypoint_mask = positive_gt_keypoints[..., 2] != 0
            else:
                keypoint_mask = torch.ones_like(
                    positive_gt_keypoints[..., 0], dtype=torch.bool
                )
            positive_boxes_grid = target_boxes_grid[foreground]
            box_sizes = positive_boxes_grid[..., 2:] - positive_boxes_grid[..., :2]
            instance_area = box_sizes[..., 0:1] * box_sizes[..., 1:2]
            pose_loss = self.keypoint_loss(
                positive_pred_keypoints,
                positive_gt_keypoints,
                keypoint_mask,
                instance_area,
            )
            if self.keypoint_dimensions == 3:
                keypoint_object_loss = self.visibility_loss(
                    positive_pred_keypoints[..., 2], keypoint_mask.to(positive_pred_keypoints.dtype)
                )

        total = (
            self.box_weight * box_loss
            + self.keypoint_weight * pose_loss
            + self.visibility_weight * keypoint_object_loss
            + self.class_weight * class_loss
            + self.distribution_weight * dfl_loss
        )
        metrics = {
            "loss": total.detach(),
            "box": box_loss.detach(),
            "pose": pose_loss.detach(),
            "kobj": keypoint_object_loss.detach(),
            "class": class_loss.detach(),
            "dfl": dfl_loss.detach(),
        }
        # The referenced trainer backpropagates the per-batch sum and reports
        # detached per-batch loss items separately.
        return {"loss": total * batch_size, "metrics": metrics}

    def _prepare_targets(
        self,
        targets: Mapping[str, torch.Tensor],
        batch_size: int,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ):
        batch_indices = targets["batch_indices"].long()
        classes = targets["classes"].long()
        normalized_boxes = targets["boxes"].float()
        normalized_keypoints = targets["keypoints"].float()
        if classes.numel() and (
            torch.any(classes < 0) or torch.any(classes >= self.num_classes)
        ):
            raise ValueError("A target class id is outside the model's configured class range")
        counts = torch.bincount(batch_indices, minlength=batch_size)
        max_targets = int(counts.max().item()) if counts.numel() else 0
        device = normalized_boxes.device
        gt_labels = torch.zeros((batch_size, max_targets, 1), dtype=torch.long, device=device)
        gt_boxes = torch.zeros((batch_size, max_targets, 4), dtype=torch.float32, device=device)
        gt_keypoints = torch.zeros(
            (batch_size, max_targets, self.num_keypoints, self.keypoint_dimensions),
            dtype=torch.float32,
            device=device,
        )
        image_height = int(image_height or self.image_size)
        image_width = int(image_width or self.image_size)
        box_scale = normalized_boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        point_scale = normalized_keypoints.new_tensor([image_width, image_height])
        for batch_index in range(batch_size):
            source_indices = torch.nonzero(batch_indices == batch_index, as_tuple=False).flatten()
            count = source_indices.numel()
            if not count:
                continue
            gt_labels[batch_index, :count, 0] = classes[source_indices]
            gt_boxes[batch_index, :count] = (
                _xywh_to_xyxy(normalized_boxes[source_indices]) * box_scale
            )
            points = normalized_keypoints[source_indices].clone()
            points[..., :2] *= point_scale
            gt_keypoints[batch_index, :count] = points
        gt_mask = gt_boxes.sum(dim=-1, keepdim=True).gt(0.0)
        return gt_labels, gt_boxes, gt_keypoints, gt_mask

    def _decode_keypoints(self, raw_keypoints: torch.Tensor, anchor_points: torch.Tensor):
        decoded = raw_keypoints.clone()
        decoded[..., 0] = decoded[..., 0] * 2.0 + anchor_points[None, :, None, 0] - 0.5
        decoded[..., 1] = decoded[..., 1] * 2.0 + anchor_points[None, :, None, 1] - 0.5
        return decoded

def _boxes_to_distances(anchor_points, boxes, reg_max):
    left_top = anchor_points - boxes[..., :2]
    right_bottom = boxes[..., 2:] - anchor_points
    return torch.cat((left_top, right_bottom), dim=-1).clamp_(0, reg_max - 0.01)


# --------------------------------------------------------------------------------------------
# RTMO losses (ported from mmpose; pure torch)
# --------------------------------------------------------------------------------------------


class VariFocalLoss(nn.Module):
    """VariFocal classification loss (mmpose port).

    Positive targets carry the OKS score; the negative weight is
    ``alpha * sigmoid(x)^gamma * (1 - label) + target``.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction="mean", alpha=0.75, gamma=2.0):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean / sum / none")
        self.reduction = reduction
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def criterion(self, output, target):
        label = (target > 1e-4).to(target)
        weight = self.alpha * output.sigmoid().pow(self.gamma) * (1 - label) + target
        output = output.clip(min=-10, max=10)
        return F.binary_cross_entropy_with_logits(output, target, reduction="none") * weight

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            if target_weight is None:
                raise ValueError("target_weight is required")
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)

        loss[torch.isinf(loss)] = 0.0
        loss[torch.isnan(loss)] = 0.0

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss * self.loss_weight


class BCELoss(nn.Module):
    """Binary cross entropy with an optional target weight (mmpose port)."""

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction="mean", use_sigmoid=False):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean / sum / none")
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        self.criterion = F.binary_cross_entropy if use_sigmoid else F.binary_cross_entropy_with_logits
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        loss = self.criterion(output, target, reduction="none")
        if self.use_target_weight:
            if target_weight is None:
                raise ValueError("target_weight is required")
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss * self.loss_weight


class IoULoss(nn.Module):
    """IoU-based box regression loss (mmpose port)."""

    def __init__(self, reduction="mean", mode="log", eps=1e-16, loss_weight=1.0):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean / sum / none")
        if mode not in ("linear", "square", "log"):
            raise ValueError("mode must be linear / square / log")
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, output, target, target_weight=None):
        ious = bbox_overlaps(output, target, is_aligned=True).clamp(min=self.eps)

        if self.mode == "linear":
            loss = 1 - ious
        elif self.mode == "square":
            loss = 1 - ious.pow(2)
        else:
            loss = -ious.log()

        if target_weight is not None:
            for _ in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1 regression loss (mmpose port)."""

    def __init__(self, reduction="mean", use_target_weight=False, loss_weight=1.0):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean / sum / none")
        self.reduction = reduction
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        if self.use_target_weight:
            if target_weight is None:
                raise ValueError("target_weight is required")
            for _ in range(target.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = F.l1_loss(output * target_weight, target * target_weight, reduction=self.reduction)
        else:
            loss = F.l1_loss(output, target, reduction=self.reduction)
        return loss * self.loss_weight


class OKSLoss(nn.Module):
    """OKS keypoint loss (mmpose port).

    Args:
        sigmas: Per-keypoint sigmas; defaults to COCO-17.
        norm_target_weight (bool): Normalize weights by the visible count
            (True) instead of dividing by the keypoint count (False).
    """

    def __init__(
        self,
        sigmas=None,
        reduction="mean",
        mode="linear",
        eps=1e-8,
        norm_target_weight=False,
        loss_weight=1.0,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean / sum / none")
        if mode not in ("linear", "square", "log"):
            raise ValueError("mode must be linear / square / log")
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.norm_target_weight = norm_target_weight
        self.eps = eps
        self.register_buffer(
            "sigmas",
            torch.as_tensor(sigmas if sigmas is not None else COCO_KEYPOINT_SIGMAS, dtype=torch.float32),
        )

    def forward(self, output, target, target_weight=None, areas=None):
        dist = torch.norm(output - target, dim=-1)
        if areas is not None:
            dist = dist / areas.pow(0.5).clip(min=self.eps).unsqueeze(-1)

        sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
        dist = dist / (sigmas * 2)

        oks = torch.exp(-dist.pow(2) / 2)

        if target_weight is not None:
            if self.norm_target_weight:
                target_weight = target_weight / target_weight.sum(dim=-1, keepdims=True).clip(min=self.eps)
            else:
                target_weight = target_weight / target_weight.size(-1)
            oks = oks * target_weight
        oks = oks.sum(dim=-1)

        if self.mode == "linear":
            loss = 1 - oks
        elif self.mode == "square":
            loss = 1 - oks.pow(2)
        else:
            loss = -oks.log()

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss * self.loss_weight


class MLECCLoss(nn.Module):
    """Maximum-likelihood loss for coordinate classification (mmpose port).

    `outputs` and `targets` are lists of per-axis distributions (e.g. x and y),
    each keeping the bin dimension last; the per-axis match probabilities are
    multiplied into a joint probability.
    """

    def __init__(self, reduction="mean", mode="log", use_target_weight=False, loss_weight=1.0):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean / sum / none")
        if mode not in ("linear", "square", "log"):
            raise ValueError("mode must be linear / square / log")
        self.reduction = reduction
        self.mode = mode
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, target_weight=None):
        if len(outputs) != len(targets):
            raise ValueError("outputs and targets must have the same length")

        prob = 1.0
        for o, t in zip(outputs, targets):
            prob = prob * (o * t).sum(dim=-1)

        if self.mode == "linear":
            loss = 1.0 - prob
        elif self.mode == "square":
            loss = 1.0 - prob.pow(2)
        else:
            loss = -torch.log(prob + 1e-4)

        loss[torch.isnan(loss)] = 0.0

        if self.use_target_weight:
            if target_weight is None:
                raise ValueError("target_weight is required")
            for _ in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.flatten(1).sum(dim=1)
        elif self.reduction == "mean":
            loss = loss.flatten(1).mean(dim=1)
        return loss * self.loss_weight
