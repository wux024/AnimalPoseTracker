"""Local target assignment for pose training.

Contains two families:

- `TaskAlignedAssigner` -- the AnimalRTPose / YOLO-style task-aligned assignment;
- the RTMO family (ported from mmpose, pure torch): `MlvlPointGenerator`,
  `bbox_overlaps` + `BBoxOverlaps2D` + `PoseOKS`, and `SimOTAAssigner`.
"""

import math

import torch
import torch.nn as nn


def _no_autocast(tensor):
    """Keep an fp32 computation out of autocast.

    PyTorch refuses ``binary_cross_entropy`` while autocast is active and checks the
    autocast *state*, not the input dtype -- so casting the inputs to fp32 is not
    enough, the context has to be switched off around the call. These are target and
    cost computations inside ``torch.no_grad()``, where fp32 was the intent anyway.
    """
    return torch.autocast(device_type=tensor.device.type, enabled=False)


def aligned_complete_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return CIoU for broadcastable xyxy box tensors."""
    a = boxes_a
    b = boxes_b
    x1_a, y1_a, x2_a, y2_a = a.unbind(dim=-1)
    x1_b, y1_b, x2_b, y2_b = b.unbind(dim=-1)
    width_a, height_a = x2_a - x1_a, y2_a - y1_a + 1e-7
    width_b, height_b = x2_b - x1_b, y2_b - y1_b + 1e-7
    intersection = (
        (x2_a.minimum(x2_b) - x1_a.maximum(x1_b)).clamp_min(0)
        * (y2_a.minimum(y2_b) - y1_a.maximum(y1_b)).clamp_min(0)
    )
    union = width_a * height_a + width_b * height_b - intersection + 1e-7
    iou = intersection / union
    outer_width = x2_a.maximum(x2_b) - x1_a.minimum(x1_b)
    outer_height = y2_a.maximum(y2_b) - y1_a.minimum(y1_b)
    outer_diagonal = outer_width.square() + outer_height.square() + 1e-7
    center_distance = (
        (x2_b + x1_b - x2_a - x1_a).square()
        + (y2_b + y1_b - y2_a - y1_a).square()
    ) / 4.0
    aspect = (4.0 / math.pi**2) * (
        torch.atan(width_b / height_b) - torch.atan(width_a / height_a)
    ).square()
    with torch.no_grad():
        alpha = aspect / (1.0 - iou + aspect + 1e-7)
    return iou - (center_distance / outer_diagonal + aspect * alpha)


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

        inside = self._centers_in_boxes(anchor_centers, gt_boxes, gt_mask)
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

    def _centers_in_boxes(self, anchor_centers, gt_boxes, gt_mask):
        """Select anchor centers inside boxes after Ultralytics' minimum-stride floor."""
        center = (gt_boxes[..., :2] + gt_boxes[..., 2:]) * 0.5
        size = (gt_boxes[..., 2:] - gt_boxes[..., :2]).clamp_min(0)
        below_stride = (size < self.stride_val) & gt_mask.bool()
        size = torch.where(below_stride, torch.full_like(size, self.stride_val), size)
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


# --------------------------------------------------------------------------------------------
# RTMO assignment family (ported from mmpose; pure torch)
# --------------------------------------------------------------------------------------------

#: COCO-17 keypoint sigmas, the default the mmpose metafile carries.
COCO_KEYPOINT_SIGMAS = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)


def _pair(value):
    """int -> (int, int); a tuple / list passes through."""
    return value if isinstance(value, (tuple, list)) else (value, value)


class MlvlPointGenerator:
    """Multi-level point grid generator (mmpose MlvlPointGenerator port).

    With ``centralize_points=True`` the offset is forced to 0 and every point
    additionally gets ``+(stride - 1) / 2`` -- both are required to reproduce
    the RTMO reference anchors exactly.

    Args:
        strides: Per-level strides, ints or (w, h) pairs.
        offset (float): Base offset in stride units; ignored when centralize.
        centralize_points (bool): Center the points inside each cell instead
            of the legacy half-stride offset.
    """

    def __init__(self, strides, offset: float = 0.5, centralize_points: bool = False):
        self.strides = [_pair(stride) for stride in strides]
        self.centralize_points = centralize_points
        self.offset = offset if not centralize_points else 0.0
        self.num_levels = len(self.strides)

    def _meshgrid(self, x, y, row_major: bool = True):
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        return yy.reshape(-1), xx.reshape(-1)

    def single_level_grid_priors(
        self, featmap_size, level_idx, dtype=torch.float32, device="cpu", with_stride: bool = False
    ):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]

        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h

        if self.centralize_points:
            shift_x = shift_x + float(stride_w - 1) / 2.0
            shift_y = shift_y + float(stride_h - 1) / 2.0

        shift_x = shift_x.to(dtype)
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w_t = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            stride_h_t = shift_yy.new_full((shift_yy.shape[0],), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w_t, stride_h_t], dim=-1)
        return shifts.to(device)

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device="cpu", with_stride: bool = False):
        """Per-level priors; with_stride=True yields (x, y, stride_w, stride_h)."""
        return [
            self.single_level_grid_priors(
                featmap_sizes[i], i, dtype=dtype, device=device, with_stride=with_stride
            )
            for i in range(self.num_levels)
        ]


def _fp16_clamp(x, min_val=None, max_val=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min_val, max_val).half()
    return x.clamp(min_val, max_val)


def bbox_overlaps(bboxes1, bboxes2, mode: str = "iou", is_aligned: bool = False, eps: float = 1e-6):
    """mmdet-style box overlaps (pure torch port).

    Supports batch dimensions; ``is_aligned=True`` pairs elements 1:1 and
    returns ``(..., m)`` instead of ``(..., m, n)``.
    """
    if mode not in ("iou", "iof", "giou"):
        raise ValueError(f"Unsupported mode {mode}")
    if bboxes1.size(-1) != 4 or bboxes2.size(-1) != 4:
        raise ValueError("boxes must be xyxy with 4 values per box")

    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.unsqueeze(0)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.unsqueeze(0)
    if bboxes1.shape[:-2] != bboxes2.shape[:-2]:
        raise ValueError("leading dimensions of the two box sets must match")

    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned and rows != cols:
        raise ValueError("aligned mode needs the same number of boxes")

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = _fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1 + area2 - overlap if mode in ("iou", "giou") else area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = _fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap if mode in ("iou", "giou") else area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_tensor = union.new_tensor([eps])
    union = torch.max(union, eps_tensor)
    ious = overlap / union
    if mode in ("iou", "iof"):
        return ious

    enclose_wh = _fp16_clamp(enclosed_rb - enclosed_lt, min_val=0)
    enclose_area = torch.max(enclose_wh[..., 0] * enclose_wh[..., 1], eps_tensor)
    return ious - (enclose_area - union) / enclose_area


class BBoxOverlaps2D:
    """2D IoU calculator (mmdet's BBoxOverlaps2D surface, pure torch)."""

    def __init__(self, scale: float = 1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, bboxes1, bboxes2, mode: str = "iou", is_aligned: bool = False):
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


class PoseOKS:
    """OKS score calculator (mmpose PoseOKS port).

    Args:
        sigmas: Per-keypoint sigmas; defaults to COCO-17.
    """

    def __init__(self, sigmas=None):
        self.sigmas = torch.as_tensor(
            sigmas if sigmas is not None else COCO_KEYPOINT_SIGMAS, dtype=torch.float32
        )

    @torch.no_grad()
    def __call__(self, output, target, target_weights, areas, eps: float = 1e-8):
        dist = torch.norm(output - target, dim=-1)
        areas = areas.reshape(*((1,) * (dist.ndim - 2)), -1, 1)
        dist = dist / areas.pow(0.5).clip(min=eps)

        if self.sigmas.device != dist.device:
            self.sigmas = self.sigmas.to(dist.device)
        sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
        dist = dist / (sigmas * 2)

        target_weights = target_weights / target_weights.sum(dim=-1, keepdims=True).clip(min=eps)
        return (torch.exp(-dist.pow(2) / 2) * target_weights).sum(dim=-1)


class SimOTAAssigner:
    """SimOTA target assignment with IoU / OKS / classification costs.

    Ported from mmpose (pure torch). `pred_instances` and `gt_instances` are
    dicts with the fields described in `assign`; the return is a dict with
    ``gt_inds`` / ``labels`` / ``max_overlaps`` (and ``num_gts``).

    Args:
        center_radius (float): Radius of the center region in stride units.
        candidate_topk (int): Candidates for the dynamic-k estimate.
        iou_weight / cls_weight / oks_weight / vis_weight (float): Cost weights.
        dynamic_k_indicator (str): 'iou' or 'oks'.
        use_keypoints_for_center (bool): Derive GT centers from keypoints.
        iou_calculator / oks_calculator: Calculators (defaults provided).
    """

    INF = 100000.0
    EPS = 1.0e-7

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
        oks_weight: float = 3.0,
        vis_weight: float = 0.0,
        dynamic_k_indicator: str = "iou",
        use_keypoints_for_center: bool = False,
        iou_calculator=None,
        oks_calculator=None,
    ):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.oks_weight = oks_weight
        self.vis_weight = vis_weight
        if dynamic_k_indicator not in ("iou", "oks"):
            raise ValueError("dynamic_k_indicator must be 'iou' or 'oks'")
        self.dynamic_k_indicator = dynamic_k_indicator
        self.use_keypoints_for_center = use_keypoints_for_center

        self.iou_calculator = iou_calculator if iou_calculator is not None else BBoxOverlaps2D()
        self.oks_calculator = oks_calculator if oks_calculator is not None else PoseOKS()

    @torch.no_grad()
    def assign(self, pred_instances, gt_instances):
        """Assign priors to GT instances.

        ``pred_instances``: dict(bboxes[N,4], scores[N,C], priors[N,4],
        keypoints[N,K,2], keypoints_visible[N,K]);
        ``gt_instances``: dict(bboxes[M,4], labels[M], keypoints[M,K,2],
        keypoints_visible[M,K], areas[M]).
        """
        gt_bboxes = gt_instances["bboxes"]
        gt_labels = gt_instances["labels"]
        gt_keypoints = gt_instances["keypoints"]
        gt_keypoints_visible = gt_instances["keypoints_visible"]
        gt_areas = gt_instances["areas"]
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances["bboxes"]
        pred_scores = pred_instances["scores"]
        priors = pred_instances["priors"]
        keypoints = pred_instances["keypoints"]
        keypoints_visible = pred_instances["keypoints_visible"]
        num_bboxes = decoded_bboxes.size(0)

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            return dict(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=decoded_bboxes.new_zeros((num_bboxes,)),
                labels=decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long),
            )

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes, gt_keypoints, gt_keypoints_visible
        )
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_pred_kpts = keypoints[valid_mask]
        valid_pred_kpts_vis = keypoints_visible[valid_mask]

        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            return dict(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=decoded_bboxes.new_zeros((num_bboxes,)),
                labels=decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long),
            )

        cost_matrix = (~is_in_boxes_and_center) * self.INF

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        if self.iou_weight > 0:
            iou_cost = -torch.log(pairwise_ious + self.EPS)
            cost_matrix = cost_matrix + iou_cost * self.iou_weight

        pairwise_oks = None
        if self.oks_weight > 0 or self.dynamic_k_indicator == "oks":
            pairwise_oks = self.oks_calculator(
                valid_pred_kpts.unsqueeze(1),
                target=gt_keypoints.unsqueeze(0),
                target_weights=gt_keypoints_visible.unsqueeze(0),
                areas=gt_areas.unsqueeze(0),
            )
            oks_cost = -torch.log(pairwise_oks + self.EPS)
            cost_matrix = cost_matrix + oks_cost * self.oks_weight

        if self.cls_weight > 0:
            gt_onehot_label = (
                torch.nn.functional.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
                .float()
                .unsqueeze(0)
                .repeat(num_valid, 1, 1)
            )
            valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
            # fp32 on purpose: BCE under autocast can overflow
            with _no_autocast(valid_pred_scores):
                cls_cost = (
                    torch.nn.functional.binary_cross_entropy(
                        valid_pred_scores.to(dtype=torch.float32), gt_onehot_label, reduction="none"
                    )
                    .sum(-1)
                    .to(dtype=valid_pred_scores.dtype)
                )
            cost_matrix = cost_matrix + cls_cost * self.cls_weight

        if self.vis_weight > 0:
            valid_pred_kpts_vis = valid_pred_kpts_vis.unsqueeze(1).repeat(1, num_gt, 1)
            gt_kpt_vis = gt_keypoints_visible.unsqueeze(0).float()
            with _no_autocast(valid_pred_kpts_vis):
                vis_cost = (
                    torch.nn.functional.binary_cross_entropy(
                        valid_pred_kpts_vis.to(dtype=torch.float32),
                        gt_kpt_vis.repeat(num_valid, 1, 1),
                        reduction="none",
                    )
                    .sum(-1)
                    .to(dtype=valid_pred_kpts_vis.dtype)
                )
            cost_matrix = cost_matrix + vis_cost * self.vis_weight

        if self.dynamic_k_indicator == "iou":
            matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask
            )
        else:
            matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
                cost_matrix, pairwise_oks, num_gt, valid_mask
            )

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -self.INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious.to(max_overlaps)
        return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes, gt_keypoints=None, gt_keypoints_visible=None):
        """Whether each prior sits inside a GT box and / or its center region."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y
        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        if self.use_keypoints_for_center and gt_keypoints_visible is not None:
            gt_kpts_cts = (gt_keypoints * gt_keypoints_visible.unsqueeze(-1)).sum(dim=-2) / (
                gt_keypoints_visible.sum(dim=-1, keepdims=True).clip(min=0)
            )
            gt_kpts_cts = gt_kpts_cts.to(gt_bboxes)
            valid_mask = gt_keypoints_visible.sum(-1) > 0
            gt_cxs[valid_mask] = gt_kpts_cts[valid_mask][..., 0]
            gt_cys[valid_mask] = gt_kpts_cts[valid_mask][..., 1]

        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y
        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :]
        )
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Match each GT to its dynamic-k lowest-cost priors, resolving conflicts."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False)
            matching_matrix[pos_idx, gt_idx] = 1

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
