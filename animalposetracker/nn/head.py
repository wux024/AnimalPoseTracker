#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: head.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.1

Overview:
    Three heads live here. They are alternatives, not siblings -- a model picks one:

        1. YOLOPoseHead  anchor-based regression head, used by AnimalRTPose
        2. SimCCHead     1D coordinate classification head, used by AnimalViTPose
        3. RTMOHead      one-stage (bottom-up) head with the Dynamic Coordinate
                         Classifier, used by the AnimalRTMO variant

    ============================ 1. YOLOPoseHead ===========================
    Anchor-based pose head, implemented as a single class.

    A YOLO-Pose model outputs boxes, classes and keypoints together, so this class
    contains all three branches:
        cv2  box regression (4 * reg_max channels)
        cv3  class prediction (nc channels)
        cv4  keypoint prediction (nk channels)
        dfl  distribution integral used to decode box distances

    All of them live in one class; there is no Detect / Pose two-level inheritance.
    Inheritance only affects code organization, not state_dict: weights are keyed by
    submodule name, so merging the classes does not break weight loading.

    Forward contract (something yaml cannot express and code must pin down explicitly):
        training=True  : (preds, kpt)
                         preds = raw cat(cv2, cv3) output per scale, [P3, P4, P5]
                         kpt   = (bs, nk, num_anchors) undecoded keypoint regressions
                         -- the training-side loss depends on these shapes
        export=True    : a single tensor cat(dbox, cls, pred_kpt)
        training=False : (cat(dbox, cls, pred_kpt), (preds, kpt))

    RLE mode (rle=True, default False):
        Enables the RLE (Residual Log-likelihood Estimation) variant, where one shared feature
        extractor feeds two 1x1 heads -- coordinates and per-keypoint sigma -- plus a RealNVP
        flow model. Rationale: RLE models the residual distribution of keypoint predictions and
        in the original paper brings +12.4 mAP on MSCOCO, while the flow model and the sigma
        branch are training-only, so inference cost is unchanged.
        With rle=True the training return becomes (preds, kpt, kpt_sigma) instead of
        (preds, kpt), because the RLE loss needs the sigma maps.
        IMPORTANT: rle=True and rle=False have different parameter names
        (cv4.0.2.* vs cv4_kpts.* / cv4_sigma.*), so their checkpoints are NOT interchangeable.
        The default stays False so that existing weights remain loadable.

    Weight compatibility: submodule names (cv2 / cv3 / cv4 / dfl and their internals)
    match the training side exactly, so trained .pt checkpoints load directly via
    state_dict. Class names and inheritance are not part of the state_dict.
    The end2end branch is not implemented (unused by this project), so one2one_cv2 and
    one2one_cv3 are absent.

Notes:
    - Requires torch only.
    - legacy must stay True: the non-legacy cv3 variant (DWConv + 1x1) has different
      parameter names and shapes and cannot share weights with this project's checkpoints.

Revision History:
    - [2026/9/16] wux024: Initial file creation
    - [2026/9/16] wux024: Merged Detect and Pose into a single Pose class, dropping the intermediate layer
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GAUEncoder
from .block import DFL, RealNVP
from .conv import Conv
from .tal import dist2bbox, make_anchors
from .transformer import SinePositionalEncoding


class YOLOPoseHead(nn.Module):
    """Anchor-based pose head: boxes, classes and keypoints regressed together.

    Renamed from `Pose` because `Pose` says which task this is, not which family it belongs
    to -- and the project now has two pose heads from two different families (this one and
    SimCCHead). The class name never enters a state_dict key, so renaming it does not
    invalidate any checkpoint.

    Args:
        nc (int): Number of classes
        kpt_shape (tuple): (number of keypoints, dimensions); 2 for x,y, 3 for x,y,visible
        ch (tuple): Input channels per scale, e.g. (192, 384, 768)
    """

    # Runtime flags set by the builder / export flow; defaults match the training side
    legacy = True  # use the traditional cv3 structure (3 plain conv layers)
    end2end = False  # end2end head is not used by this project
    shape = None  # cached input shape during inference
    anchors = torch.empty(0)  # cached anchors during inference
    strides = torch.empty(0)  # cached strides during inference
    export = False  # export-mode flag
    format = "torchscript"  # export-format flag
    dynamic = False  # dynamic-axes flag

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=(), rle=False):
        super().__init__()
        if not self.legacy:
            # Upstream cv3 has two variants: the non-legacy one combines DWConv with a 1x1 conv,
            # which produces different parameter names and shapes, so weights cannot be shared.
            # Fail loudly here instead of silently building a mismatched structure that would only
            # blow up later at load_state_dict time.
            raise NotImplementedError(
                "Non-legacy cv3 structure is not implemented (all checkpoints in this project are "
                "legacy: 3 plain conv layers). To support it, port the else branch from the "
                "training-side nn/modules/head.py."
            )

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor (excluding keypoints)
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]  # total number of keypoint regressions

        # Box branch and class branch
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # Keypoint branch.
        # rle=False (default): one 3-layer branch, parameter-compatible with existing checkpoints.
        # rle=True: RLE (Residual Log-likelihood Estimation) variant -- a single shared feature
        # extractor feeding two 1x1 heads (coordinates and per-keypoint sigma), plus a RealNVP
        # flow model. The flow model is consumed by the RLE loss during training only and plays no
        # part in inference, so it adds no inference cost.
        # NOTE: rle=True checkpoints are NOT interchangeable with rle=False ones (different
        # parameter names), so the default stays False to keep existing weights loadable.
        self.rle = rle
        if rle:
            self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y per keypoint
            self.flow_model = RealNVP()
            c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
            self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)
            self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
            self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)
        else:
            c4 = max(ch[0] // 4, self.nk)
            self.cv4 = nn.ModuleList(
                nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch
            )

    def forward(self, x):
        """Return values per the forward contract; shapes vary with training / export mode."""
        bs = x[0].shape[0]  # batch size

        # Keypoint branch: concatenate all scales along the anchor dimension.
        # In RLE mode the shared feature extractor runs once and feeds both heads; the sigma head
        # is only evaluated during training, because it exists purely to feed the RLE loss.
        if self.rle:
            feats = [self.cv4[i](x[i]) for i in range(self.nl)]
            kpt = torch.cat([self.cv4_kpts[i](feats[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
            kpt_sigma = (
                torch.cat([self.cv4_sigma[i](feats[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], -1)
                if self.training
                else None
            )
        else:
            kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, nk, h*w)
            kpt_sigma = None

        # Box + class branch: concatenate in place into (bs, no, h*w)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            # RLE mode additionally returns the sigma maps so the RLE loss can consume them
            return (x, kpt, kpt_sigma) if self.rle else (x, kpt)

        pred = self._inference(x, kpt, bs)
        if self.export:
            return pred
        return pred, (x, kpt)

    def _inference(self, x, kpt, bs):
        """Decode into cat(dbox, cls, pred_kpt), shaped (bs, 4 + nc + nk, num_anchors)."""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        cache_device = x[0].device
        cache_dtype = x[0].dtype
        cache_mismatch = (
            self.anchors.device != cache_device
            or self.strides.device != cache_device
            or self.anchors.dtype != cache_dtype
            or self.strides.dtype != cache_dtype
        )
        if self.dynamic or self.shape != shape or cache_mismatch:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid(), self.kpts_decode(bs, kpt)), 1)

    def bias_init(self):
        """Initialize biases to suppress the high-confidence prediction storm early in training."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """(ltrb distances, anchors) -> boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)

    def kpts_decode(self, bs, kpts):
        """Decode keypoint regressions into image pixel coordinates (export mode uses a normalized grid)."""
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class _SimCCDeconvHead(nn.Module):
    """Deconvolution stack plus a 1x1 output convolution.

    Mirrors mmpose's `HeatmapHead`, which `SimCCHead` instantiates as `self.deconv_head`.
    Keeping both the nesting and the layer names means a SimCCHead checkpoint produced by
    mmpose loads here without any key remapping.

    Every deconvolution doubles the spatial size, so the feature map grows by 2 ** N.
    `nn.Sequential` is fed the layers flat (not nested), which is what makes the keys come
    out as `deconv_layers.0` / `deconv_layers.1` / `deconv_layers.3` ... -- exactly as in
    mmpose, where the BatchNorm and ReLU are siblings of the convolution, not children.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: Sequence[int] = (256,),
        deconv_kernel_sizes: Sequence[int] = (4,),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if len(deconv_out_channels) != len(deconv_kernel_sizes):
            raise ValueError(
                f"deconv_out_channels and deconv_kernel_sizes must have the same length, "
                f"got {deconv_out_channels} and {deconv_kernel_sizes}"
            )

        layers: List[nn.Module] = []
        c_in = in_channels
        for c_out, kernel in zip(deconv_out_channels, deconv_kernel_sizes):
            # padding / output_padding chosen so that stride 2 exactly doubles the resolution
            if kernel == 4:
                padding, output_padding = 1, 0
            elif kernel == 3:
                padding, output_padding = 1, 1
            elif kernel == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(f"Unsupported deconv kernel size {kernel}, expected 2, 3 or 4")
            layers.append(
                nn.ConvTranspose2d(
                    c_in,
                    c_out,
                    kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,  # followed by BatchNorm, so a bias would be redundant
                )
            )
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.deconv_layers = nn.Sequential(*layers)

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError("conv_out_channels and conv_kernel_sizes must match in length")
            convs: List[nn.Module] = []
            for c_out, kernel in zip(conv_out_channels, conv_kernel_sizes):
                convs.append(nn.Conv2d(c_in, c_out, kernel, stride=1, padding=(kernel - 1) // 2))
                convs.append(nn.BatchNorm2d(c_out))
                convs.append(nn.ReLU(inplace=True))
                c_in = c_out
            self.conv_layers = nn.Sequential(*convs)
        else:
            self.conv_layers = nn.Identity()

        self.final_layer = nn.Conv2d(c_in, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the feature map to heatmap resolution and project it to out_channels."""
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        return self.final_layer(x)


class SimCCHead(nn.Module):
    """SimCC head: keypoint localisation as two 1D coordinate classifications.

    Proposed in `SimCC <https://arxiv.org/abs/2107.03332>`_ (Li et al., 2022). Instead of
    regressing a 2D heatmap, each keypoint is classified separately along the x and the y
    axis, so one keypoint becomes two 1D distributions. That removes the 2D quantisation
    error that heatmap heads suffer from, and makes the whole head cheap: a deconvolution
    stack followed by two linear layers.

    Sub-module layout mirrors mmpose's `SimCCHead`, in particular the `deconv_head` nesting,
    so an existing mmpose checkpoint loads without key remapping.

    Args:
        in_channels (int): Channels of the backbone feature map.
        out_channels (int): Number of keypoints.
        input_size (tuple): Model input size as (w, h) in pixels.
        in_featuremap_size (tuple): Spatial size of the backbone feature map as (h, w).
        simcc_split_ratio (float): Pixel splitting ratio of the 1D representation.
        deconv_out_channels (sequence[int]): Output channels of each deconvolution.
        deconv_kernel_sizes (sequence[int]): Kernel size of each deconvolution; 2, 3 or 4.
        conv_out_channels (sequence[int], optional): Extra convolution stack inserted before
            the 1x1 output convolution. None means no extra stack.
        conv_kernel_sizes (sequence[int], optional): Kernel sizes of that extra stack.

    Shape bookkeeping (easy to get wrong, so it is spelled out):
        heatmap_size = in_featuremap_size * 2 ** len(deconv_out_channels)
        flatten_dims = heatmap_size[0] * heatmap_size[1]   <- width of the Linear layers
        W = int(input_size[0] * simcc_split_ratio)         <- length of the x distribution
        H = int(input_size[1] * simcc_split_ratio)         <- length of the y distribution

    Note the Linear layers act on the flattened spatial dimensions, so the output is
    (B, out_channels, W) and (B, out_channels, H): one 1D distribution per keypoint per axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        deconv_out_channels: Sequence[int] = (256,),
        deconv_kernel_sizes: Sequence[int] = (4,),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if len(deconv_out_channels) == 0:
            raise ValueError("deconv_out_channels must not be empty for SimCCHead")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = tuple(input_size)
        self.in_featuremap_size = tuple(in_featuremap_size)
        self.simcc_split_ratio = simcc_split_ratio

        num_deconv = len(deconv_out_channels)
        self.heatmap_size = tuple(s * (2**num_deconv) for s in self.in_featuremap_size)

        self.deconv_head = _SimCCDeconvHead(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
        )

        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]
        w = int(self.input_size[0] * self.simcc_split_ratio)
        h = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = nn.Linear(flatten_dims, w)
        self.mlp_head_y = nn.Linear(flatten_dims, h)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize as in mmpose: small normal weights for the convolutions and the Linear
        layers, unit weights for the BatchNorm running statistics."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor | list | tuple): Backbone feature map. A list or tuple is
                accepted for symmetry with the Pose head and the mmpose convention, in which
                the head receives a list of multi-scale features; the last one is used.

        Returns:
            tuple: (pred_x with shape (B, out_channels, W), pred_y with shape (B, out_channels, H))
        """
        if isinstance(x, (list, tuple)):
            x = x[-1]

        feats = self.deconv_head(x)
        feats = torch.flatten(feats, 2)  # (B, out_channels, heatmap_h * heatmap_w)
        return self.mlp_head_x(feats), self.mlp_head_y(feats)


# --------------------------------------------------------------------------------------------
# RTMO head (ported from mmpose; pure torch, no mmcv)
# --------------------------------------------------------------------------------------------


class _ScaleConst(nn.Module):
    """Multiply by a constant (stand-in for mmcv's Scale).

    mmcv's `Scale(0.1)` carries a learnable factor, but the RTMO configs never
    train it, so a plain constant keeps behaviour and drops the parameter.
    """

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class RTMOHeadModule(nn.Module):
    """One-stage pose head module: classification plus a pose/regression branch.

    Ported from mmpose's RTMO (`rtmo_head.py`). Sub-module names (`conv_cls` /
    `conv_pose` / `out_cls` / `out_bbox` / `out_kpt_reg` / `out_kpt_vis` /
    `out_pose`) are kept identical so reference checkpoints load.

    Args:
        num_keypoints (int): Number of keypoints for one instance.
        in_channels (int): Channels of the input feature maps.
        num_classes (int): Classes excluding background. Defaults to 1.
        widen_factor (float): Width multiplier. Defaults to 1.0.
        cls_feat_channels (int): Channels of the classification branch.
        stacked_convs (int): Stacked convolutions per branch.
        num_groups (int): Groups of the grouped convolutions in the pose branch.
        channels_per_group (int): Channels per group in the pose branch.
        pose_vec_channels (int): Channels of the pose feature; -1 keeps the
            regression features instead of a separate projection.
        featmap_strides (Sequence[int]): Strides of the input feature maps.
    """

    def __init__(
        self,
        num_keypoints,
        in_channels,
        num_classes=1,
        widen_factor=1.0,
        cls_feat_channels=256,
        stacked_convs=2,
        num_groups=8,
        channels_per_group=36,
        pose_vec_channels=-1,
        featmap_strides=(8, 16, 32),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_feat_channels = int(cls_feat_channels * widen_factor)
        self.stacked_convs = stacked_convs

        self.featmap_strides = featmap_strides
        self.in_channels = int(in_channels * widen_factor)
        self.num_keypoints = num_keypoints

        self.num_groups = num_groups
        self.channels_per_group = int(widen_factor * channels_per_group)
        self.pose_vec_channels = pose_vec_channels

        self._init_layers()

    def _init_layers(self):
        self._init_cls_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Classification branch: stacked 3x3 convolutions + a 1x1 output."""
        self.conv_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.cls_feat_channels
                stacked_convs.append(Conv(chn, self.cls_feat_channels, 3, p=1))
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        self.out_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(nn.Conv2d(self.cls_feat_channels, self.num_classes, 1))

    def _init_pose_branch(self):
        """Pose branch: grouped convolutions + box / kpt / visibility / pose outputs."""
        self.conv_pose = nn.ModuleList()
        out_chn = self.num_groups * self.channels_per_group
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                chn = self.in_channels if i == 0 else out_chn
                groups = 1 if i == 0 else self.num_groups
                stacked_convs.append(Conv(chn, out_chn, 3, p=1, g=groups))
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        self.out_bbox = nn.ModuleList()
        self.out_kpt_reg = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(out_chn, 4, 1))
            self.out_kpt_reg.append(nn.Conv2d(out_chn, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(out_chn, self.num_keypoints, 1))

        if self.pose_vec_channels > 0:
            self.out_pose = nn.ModuleList()
            for _ in self.featmap_strides:
                self.out_pose.append(nn.Conv2d(out_chn, self.pose_vec_channels, 1))

    def init_weights(self):
        """Prior-initialize the classification bias for stable training."""
        bias_init = -math.log((1 - 0.01) / 0.01)
        for conv_cls in self.out_cls:
            conv_cls.bias.data.fill_(bias_init)

    def forward(self, x):
        """Run every branch on the split features of each level.

        Returns (cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats), each a
        per-level list. `kpt_offsets` is only produced while training (it feeds the
        proxy poses for positive-sample selection).
        """
        cls_scores, bbox_preds = [], []
        kpt_offsets, kpt_vis = [], []
        pose_feats = []

        for i in range(len(x)):
            cls_feat, reg_feat = x[i].split(x[i].size(1) // 2, 1)

            cls_feat = self.conv_cls[i](cls_feat)
            reg_feat = self.conv_pose[i](reg_feat)

            cls_scores.append(self.out_cls[i](cls_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            if self.training:
                # `kpt_offsets` generates the proxy poses for positive sample
                # selection during training
                kpt_offsets.append(self.out_kpt_reg[i](reg_feat))
            kpt_vis.append(self.out_kpt_vis[i](reg_feat))

            if self.pose_vec_channels > 0:
                pose_feats.append(self.out_pose[i](reg_feat))
            else:
                pose_feats.append(reg_feat)

        return cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats


class DCC(nn.Module):
    """Dynamic Coordinate Classifier for one-stage pose estimation.

    Ported from mmpose's RTMO. Turns the pose feature of every positive grid into
    per-keypoint x / y distributions over dynamically allocated bins, and decodes
    them with an integral (expectation) instead of an argmax.

    Args:
        in_channels (int): Channels of the pose feature.
        num_keypoints (int): Number of keypoints.
        feat_channels (int): Internal feature channels.
        num_bins (Tuple[int, int]): Bin counts for x and y.
        spe_channels (int): Sine positional encoding channels. Defaults to 128.
        spe_temperature (float): Sine positional encoding temperature. Defaults to 300.
        gau_cfg (dict, optional): Config for the gated attention unit.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        feat_channels: int,
        num_bins,
        spe_channels: int = 128,
        spe_temperature: float = 300.0,
        gau_cfg=None,
    ):
        super().__init__()
        if gau_cfg is None:
            gau_cfg = dict(
                s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn="SiLU", pos_enc="add"
            )

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.feat_channels = feat_channels
        self.num_bins = num_bins
        self.gau_cfg = gau_cfg

        self.spe = SinePositionalEncoding(out_channels=spe_channels, temperature=spe_temperature)
        self.spe_feat_channels = spe_channels

        self._build_layers()
        self._build_basic_bins()

    def _build_layers(self):
        if self.gau_cfg is not None:
            gau_cfg = dict(self.gau_cfg)
            gau_cfg["in_token_dims"] = self.feat_channels
            gau_cfg["out_token_dims"] = self.feat_channels
            self.gau = GAUEncoder(**gau_cfg)
            if gau_cfg.get("pos_enc", "none") in ("add", "rope"):
                self.pos_enc = nn.Parameter(torch.randn(self.num_keypoints, gau_cfg["s"]))

        # fully-connected layers to convert pose feats to keypoint feats
        self.pose_to_kpts = nn.Sequential(
            nn.Linear(self.in_channels, self.feat_channels * self.num_keypoints),
            nn.BatchNorm1d(self.feat_channels * self.num_keypoints),
        )

        # adapter layers for dynamic encodings
        self.x_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)
        self.y_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)

        # fully-connected layers to predict sigma
        self.sigma_fc = nn.Sequential(
            nn.Linear(self.in_channels, self.num_keypoints),
            nn.Sigmoid(),
            _ScaleConst(0.1),
        )

    def _build_basic_bins(self):
        """Basic bin coordinates for x and y, spanning [-0.5, 0.5]."""
        self.register_buffer("y_bins", torch.linspace(-0.5, 0.5, self.num_bins[1]))
        self.register_buffer("x_bins", torch.linspace(-0.5, 0.5, self.num_bins[0]))

    def _apply_softmax(self, x_hms, y_hms):
        """Numerically stable softmax over the bin axis of the 1-D heatmaps."""
        x_hms = x_hms.clamp(min=-5e4, max=5e4)
        y_hms = y_hms.clamp(min=-5e4, max=5e4)
        pred_x = x_hms - x_hms.max(dim=-1, keepdims=True).values.detach()
        pred_y = y_hms - y_hms.max(dim=-1, keepdims=True).values.detach()

        exp_x, exp_y = pred_x.exp(), pred_y.exp()
        prob_x = exp_x / (exp_x.sum(dim=-1, keepdims=True) + 1e-8)
        prob_y = exp_y / (exp_y.sum(dim=-1, keepdims=True) + 1e-8)
        return prob_x, prob_y

    def _get_bin_enc(self, bbox_cs, grids):
        """Dynamic bin allocation + sine encoding for the x / y bins."""
        center, scale = bbox_cs.split(2, dim=-1)
        center = center - grids

        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 1:2] + center[..., 1:2]

        x_bins_enc = self.x_fc(self.spe(position=x_bins))
        y_bins_enc = self.y_fc(self.spe(position=y_bins))
        return x_bins_enc, y_bins_enc

    def _pose_feats_to_heatmaps(self, pose_feats, x_bins_enc, y_bins_enc):
        """Pose features -> per-keypoint features -> x / y 1-D heatmaps."""
        kpt_feats = self.pose_to_kpts(pose_feats)
        kpt_feats = kpt_feats.reshape(*kpt_feats.shape[:-1], self.num_keypoints, self.feat_channels)

        if hasattr(self, "gau"):
            kpt_feats = self.gau(kpt_feats, pos_enc=getattr(self, "pos_enc", None))

        x_hms = torch.matmul(kpt_feats, x_bins_enc.transpose(-1, -2).contiguous())
        y_hms = torch.matmul(kpt_feats, y_bins_enc.transpose(-1, -2).contiguous())
        return x_hms, y_hms

    def _decode_xy_heatmaps(self, x_hms, y_hms, bbox_cs):
        """Integral (expectation) over the dynamic bins."""
        center, scale = bbox_cs.split(2, dim=-1)

        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 1:2] + center[..., 1:2]

        x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
        y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)
        return torch.stack((x, y), dim=-1)

    def generate_target_heatmap(self, kpt_targets, bbox_cs, sigmas, areas):
        """Laplacian-error targets for the 1-D heatmaps."""
        center, scale = bbox_cs.split(2, dim=-1)
        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 1:2] + center[..., 1:2]

        dist_x = torch.abs(kpt_targets.narrow(2, 0, 1) - x_bins.unsqueeze(1))
        dist_y = torch.abs(kpt_targets.narrow(2, 1, 1) - y_bins.unsqueeze(1))

        areas = areas.pow(0.5).clip(min=1).reshape(-1, 1, 1)
        sigmas = sigmas.clip(min=1e-3).unsqueeze(2)
        dist_x = dist_x / areas / sigmas
        dist_y = dist_y / areas / sigmas

        hm_x = torch.exp(-dist_x / 2) / sigmas
        hm_y = torch.exp(-dist_y / 2) / sigmas
        return hm_x, hm_y

    def forward_train(self, pose_feats, bbox_cs, grids):
        """Training forward: returns (pose predictions, heatmaps, sigmas)."""
        sigmas = self.sigma_fc(pose_feats)
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc, y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        return pose_preds, (x_hms, y_hms), sigmas

    @torch.no_grad()
    def forward_test(self, pose_feats, bbox_cs, grids):
        """Inference forward: returns the decoded pose predictions."""
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc, y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        return self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)

    def switch_to_deploy(self, test_cfg=None):
        """Fold the BN into the FC and freeze the GAU path (inference optimizations)."""
        if getattr(self, "deploy", False):
            return
        self._convert_pose_to_kpts()
        if hasattr(self, "gau"):
            self._convert_gau()
        self.deploy = True

    def _convert_pose_to_kpts(self):
        """Fold the BatchNorm1d of `pose_to_kpts` into the preceding Linear."""
        fc, bn = self.pose_to_kpts
        std = (bn.running_var + bn.eps).sqrt()
        weight = fc.weight * (bn.weight / std).unsqueeze(1)
        bias = bn.bias + (fc.bias - bn.running_mean) * bn.weight / std
        fc.weight.data = weight.detach()
        fc.bias.data = bias.detach()
        self.pose_to_kpts = fc

    def _convert_gau(self):
        """Pre-expand gamma / beta of the GAU and rewrite its forward (export-friendly)."""
        gamma_q = self.gau.gamma[0].view(1, 1, 1, self.gau.gamma.size(-1))
        gamma_k = self.gau.gamma[1].view(1, 1, 1, self.gau.gamma.size(-1))
        beta_q = self.gau.beta[0].view(1, 1, 1, self.gau.beta.size(-1))
        beta_k = self.gau.beta[1].view(1, 1, 1, self.gau.beta.size(-1))

        if hasattr(self, "pos_enc"):
            pos_enc = self.pos_enc.reshape(1, 1, *self.pos_enc.shape)
            beta_q = beta_q + pos_enc
            beta_k = beta_k + pos_enc

        gamma_q, gamma_k = gamma_q.detach(), gamma_k.detach()
        beta_q, beta_k = beta_q.detach(), beta_k.detach()

        @torch.no_grad()
        def _forward(self, x, *args, **kwargs):
            norm = torch.linalg.norm(x, dim=-1, keepdim=True) * self.ln.scale
            x = x / norm.clamp(min=self.ln.eps) * self.ln.g

            uv = self.uv(x)
            uv = self.act_fn(uv)

            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
            q = base * gamma_q.to(base) + beta_q.to(base)
            k = base * gamma_k.to(base) + beta_k.to(base)
            qk = torch.matmul(q, k.transpose(-1, -2))

            kernel = torch.square(torch.nn.functional.relu(qk / self.sqrt_s))
            x = u * torch.matmul(kernel, v)
            return self.o(x)

        self.gau._forward = _forward.__get__(self.gau)


def _bbox_xyxy2cs(bbox, padding: float = 1.0):
    """(left, top, right, bottom) -> (center, scale); torch tensors supported."""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def _filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter by score threshold, then keep the topk (mmpose helper port)."""
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, (list, tuple)):
            filtered_results = [result[keep_idxs] for result in results]
        elif torch.is_tensor(results):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError("Only supports dict or list or Tensor")
    return scores, labels, keep_idxs, filtered_results


def _nms_torch(boxes, scores, threshold):
    """Greedy NMS for torch tensors; returns the kept indices."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(0, descending=True)

    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        xx1 = torch.maximum(x1[current], x1[remaining])
        yy1 = torch.maximum(y1[current], y1[remaining])
        xx2 = torch.minimum(x2[current], x2[remaining])
        yy2 = torch.minimum(y2[current], y2[remaining])
        inter = (xx2 - xx1).clamp_(min=0) * (yy2 - yy1).clamp_(min=0)
        if inter.sum() == 0:
            order = remaining
            continue
        overlap = inter / (areas[current] + areas[remaining] - inter)
        order = remaining[overlap <= threshold]

    return torch.stack(keep)


class RTMOHead(nn.Module):
    """One-stage (bottom-up) pose head with the Dynamic Coordinate Classifier.

    Ported from mmpose's RTMO. Unlike the other two heads this one is
    detector-style: every positive grid of the input feature maps predicts a box
    and a full pose at once, so no detector and no instance crop are involved.

    Dependencies are injected as plain objects, keeping this file torch-only:

    - `prior_generator`: exposes ``grid_priors(featmap_sizes, dtype, device,
      with_stride)``;
    - `assigner`: exposes ``assign(pred_instances: dict, gt_instances: dict) ->
      dict`` returning ``gt_inds`` / ``labels`` / ``max_overlaps``, plus
      ``oks_calculator(pred_kpts, gt_kpts, vis, areas)``;
    - losses: callables; `loss_mle` is read for its ``loss_weight`` attribute.

    Data samples are dicts: ``{"gt_instance_labels": {"bboxes", "labels",
    "keypoints", "keypoints_visible", "areas"}, "gt_fields": {...},
    "metainfo": {"input_size": (H, W)}}``.
    """

    def __init__(
        self,
        num_keypoints: int,
        head_module_cfg: dict,
        featmap_strides=(16, 32),
        num_classes: int = 1,
        use_aux_loss: bool = False,
        proxy_target_cc: bool = False,
        assigner=None,
        prior_generator=None,
        bbox_padding: float = 1.25,
        overlaps_power: float = 1.0,
        dcc_cfg=None,
        loss_cls=None,
        loss_bbox=None,
        loss_oks=None,
        loss_vis=None,
        loss_mle=None,
        loss_bbox_aux=None,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.featmap_strides = tuple(featmap_strides)
        self.use_aux_loss = use_aux_loss
        self.overlaps_power = overlaps_power
        self.prior_generator = prior_generator
        self.assigner = assigner
        self.bbox_padding = bbox_padding

        head_module_cfg = dict(head_module_cfg)
        head_module_cfg["featmap_strides"] = list(featmap_strides)
        head_module_cfg["num_keypoints"] = num_keypoints
        self.head_module = RTMOHeadModule(**head_module_cfg)

        self.proxy_target_cc = proxy_target_cc
        if dcc_cfg is not None:
            dcc_cfg = dict(dcc_cfg)
            dcc_cfg["num_keypoints"] = num_keypoints
            self.dcc = DCC(**dcc_cfg)

        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_oks = loss_oks
        self.loss_vis = loss_vis
        self.loss_mle = loss_mle
        self.loss_bbox_aux = loss_bbox_aux

        # inference-time prior cache (filled by predict / switch_to_deploy)
        self.featmap_sizes = None
        self.mlvl_priors = None
        self.flatten_priors = None
        self.flatten_stride = None

    def forward(self, feats):
        return self.head_module(feats)

    def init_weights(self):
        self.head_module.init_weights()

    # ------------------------------------------------------------------
    # decoding helpers (ported from YOLOXPoseHead)
    # ------------------------------------------------------------------

    def decode_bbox(self, pred_bboxes, priors, stride):
        """Distance predictions -> xyxy boxes, relative to each prior."""
        stride = stride.view(1, -1, 1)
        priors = priors.view(1, -1, 2)
        xys = (pred_bboxes[..., :2] * stride) + priors
        whs = pred_bboxes[..., 2:].exp() * stride
        tl = xys - (whs * 0.5)
        br = xys + (whs * 0.5)
        return torch.cat([tl, br], dim=-1)

    def decode_kpt_reg(self, pred_kpt_offsets, priors, stride):
        """Keypoint offset predictions -> absolute keypoints."""
        shape = pred_kpt_offsets.shape
        stride = stride.view(1, -1, 1, 1)
        priors = priors.view(1, -1, 1, 2)
        pred_kpt_offsets = pred_kpt_offsets.reshape(*shape[:-1], self.num_keypoints, 2)
        return (pred_kpt_offsets * stride) + priors

    def _flatten_predictions(self, preds):
        """Per-level (B, C, H, W) list -> a single (B, sum(HW), C) tensor."""
        if len(preds) == 0:
            return None
        preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)

    # ------------------------------------------------------------------
    # target generation (ported from YOLOXPoseHead, dict-based samples)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_targets(
        self,
        flat_priors,
        cls_scores,
        objectness,
        bbox_preds,
        kpt_reg_preds,
        kpt_vis,
        batch_data_samples,
    ):
        """Per-image targets, concatenated across the batch."""
        num_imgs = len(batch_data_samples)

        # normalize the predictions before assignment, guarding against nan
        batch_cls_scores = cls_scores.clip(-1e4, 1e4).sigmoid()
        batch_objectness = objectness.clip(-1e4, 1e4).sigmoid()
        batch_kpt_vis = kpt_vis.clip(-1e4, 1e4).sigmoid()
        batch_cls_scores[torch.isnan(batch_cls_scores)] = 0
        batch_objectness[torch.isnan(batch_objectness)] = 0

        targets_each = []
        for i in range(num_imgs):
            targets_each.append(
                self._get_targets_single(
                    flat_priors,
                    batch_cls_scores[i],
                    batch_objectness[i],
                    bbox_preds[i],
                    kpt_reg_preds[i],
                    batch_kpt_vis[i],
                    batch_data_samples[i],
                )
            )

        targets = list(zip(*targets_each))
        for i in range(len(targets)):
            if torch.is_tensor(targets[i][0]):
                non_empty = [target for target in targets[i] if target.size(0) > 0]
                targets[i] = torch.cat(non_empty) if non_empty else targets[i][0].new_zeros((0,))

        (foreground_masks, cls_targets, obj_targets, obj_weights, bbox_targets,
         kpt_targets, vis_targets, vis_weights, pos_areas, pos_priors,
         group_indices, num_fg_imgs) = targets

        # auxiliary regression targets (relative to their priors)
        if self.use_aux_loss:
            pos_wh = pos_priors[:, -2:]
            pos_xy = pos_priors[:, :2]
            bbox_cxcy = (bbox_targets[:, :2] + bbox_targets[:, 2:]) / 2
            bbox_wh = bbox_targets[:, 2:] - bbox_targets[:, :2]
            bbox_aux_targets = torch.cat(
                [(bbox_cxcy - pos_xy) / pos_wh, (bbox_wh / pos_wh + 1e-8).log()], dim=-1
            )
            kpt_aux_targets = (kpt_targets - pos_xy[:, None, :]) / (pos_wh[:, None, :])
        else:
            bbox_aux_targets = None
            kpt_aux_targets = None

        return (
            foreground_masks, cls_targets, obj_targets, obj_weights,
            bbox_targets, bbox_aux_targets, kpt_targets, kpt_aux_targets,
            vis_targets, vis_weights, pos_areas, pos_priors, group_indices,
            num_fg_imgs,
        )

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors,
        cls_scores,
        objectness,
        decoded_bboxes,
        decoded_kpts,
        kpt_vis,
        data_sample,
    ):
        """Targets for one image (see `_get_targets`)."""
        num_priors = priors.size(0)
        gt_instances = data_sample["gt_instance_labels"]
        gt_fields = data_sample.get("gt_fields", {})
        num_gts = len(gt_instances["bboxes"])

        # no GT: everything empty, objectness weight 1 everywhere
        if num_gts == 0:
            return (
                cls_scores.new_zeros(num_priors).bool(),
                cls_scores.new_zeros((0, self.num_classes)),
                cls_scores.new_zeros((num_priors, 1)),
                cls_scores.new_ones((num_priors, 1)),
                cls_scores.new_zeros((0, 4)),
                cls_scores.new_zeros((0, self.num_keypoints, 2)),
                cls_scores.new_zeros((0, self.num_keypoints)),
                cls_scores.new_zeros((0, self.num_keypoints)),
                cls_scores.new_zeros((0,)),
                priors[:0],
                [],
                0,
            )

        # assign positive samples
        scores = cls_scores * objectness
        pred_instances = dict(
            bboxes=decoded_bboxes,
            scores=scores.sqrt_(),
            priors=priors,
            keypoints=decoded_kpts,
            keypoints_visible=kpt_vis,
        )
        assign_result = self.assigner.assign(pred_instances=pred_instances, gt_instances=gt_instances)

        pos_inds = torch.nonzero(assign_result["gt_inds"] > 0, as_tuple=False).squeeze(-1).unique()
        num_pos_per_img = pos_inds.size(0)
        pos_gt_labels = assign_result["labels"][pos_inds]
        pos_assigned_gt_inds = assign_result["gt_inds"][pos_inds] - 1

        bbox_target = gt_instances["bboxes"][pos_assigned_gt_inds.long()]

        max_overlaps = assign_result["max_overlaps"][pos_inds]
        cls_target = F.one_hot(pos_gt_labels, self.num_classes) * max_overlaps.unsqueeze(-1)

        kpt_target = gt_instances["keypoints"][pos_assigned_gt_inds]
        vis_target = gt_instances["keypoints_visible"][pos_assigned_gt_inds]
        if "keypoints_visible_weights" in gt_instances:
            vis_weight = gt_instances["keypoints_visible_weights"][pos_assigned_gt_inds]
        else:
            vis_weight = vis_target.new_ones(vis_target.shape)
        pos_areas = gt_instances["areas"][pos_assigned_gt_inds]

        obj_target = torch.zeros_like(objectness)
        obj_target[pos_inds] = 1

        # down-weight the grids that predict unlabeled instances
        invalid_mask = gt_fields.get("heatmap_mask", None)
        if invalid_mask is not None and (invalid_mask != 0.0).any():
            pred_vis = (kpt_vis.unsqueeze(-1) > 0.3).float()
            mean_kpts = (decoded_kpts * pred_vis).sum(dim=1) / pred_vis.sum(dim=1).clamp(min=1e-8)
            mean_kpts = mean_kpts.reshape(1, -1, 1, 2)
            wh = invalid_mask.shape[-1]
            grids = mean_kpts / (wh - 1) * 2 - 1
            mask = invalid_mask.unsqueeze(0).float()
            weight = F.grid_sample(mask, grids, mode="bilinear", padding_mode="zeros")
            obj_weight = 1.0 - weight.reshape(num_priors, 1)
        else:
            obj_weight = obj_target.new_ones(obj_target.shape)

        foreground_mask = torch.zeros_like(objectness.squeeze()).to(torch.bool)
        foreground_mask[pos_inds] = 1
        pos_priors = priors[pos_inds]
        group_index = [
            torch.where(pos_assigned_gt_inds == num)[0] for num in torch.unique(pos_assigned_gt_inds)
        ]

        return (
            foreground_mask, cls_target, obj_target, obj_weight, bbox_target,
            kpt_target, vis_target, vis_weight, pos_areas, pos_priors,
            group_index, num_pos_per_img,
        )

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------

    def _reduce_mean(self, value):
        """Cross-rank mean under DDP, a no-op on a single process."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = value.clone()
            torch.distributed.all_reduce(value)
            value /= torch.distributed.get_world_size()
        return value

    def loss(self, feats, batch_data_samples, train_cfg=None):
        """Compute the RTMO losses for one batch. Returns a dict."""
        cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_vecs = self.forward(feats)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )
        flatten_priors = torch.cat(mlvl_priors)

        flatten_cls_scores = self._flatten_predictions(cls_scores)
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        # RTMO has no objectness branch: a constant that sigmoids to ~1 stands in
        flatten_objectness = torch.ones_like(flatten_cls_scores).detach().narrow(-1, 0, 1) * 1e4
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)

        flatten_bbox_decoded = self.decode_bbox(flatten_bbox_preds, flatten_priors[..., :2], flatten_priors[..., -1])
        flatten_kpt_decoded = self.decode_kpt_reg(
            flatten_kpt_offsets, flatten_priors[..., :2], flatten_priors[..., -1]
        )

        targets = self._get_targets(
            flatten_priors,
            flatten_cls_scores.detach(),
            flatten_objectness.detach(),
            flatten_bbox_decoded.detach(),
            flatten_kpt_decoded.detach(),
            flatten_kpt_vis.detach(),
            batch_data_samples,
        )
        (pos_masks, cls_targets, obj_targets, obj_weights, bbox_targets, bbox_aux_targets,
         kpt_targets, kpt_aux_targets, vis_targets, vis_weights, pos_areas, pos_priors,
         group_indices, num_fg_imgs) = targets

        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_scores.device)
        num_total_samples = max(self._reduce_mean(num_pos), 1.0)

        extra_info = dict(num_samples=num_total_samples)
        losses = dict()
        cls_preds_all = flatten_cls_scores.view(-1, self.num_classes)

        if num_pos > 0:
            # bbox
            bbox_preds_pos = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses["loss_bbox"] = self.loss_bbox(bbox_preds_pos, bbox_targets) / num_total_samples

            if self.use_aux_loss and self.loss_bbox_aux is not None:
                bbox_preds_raw = flatten_bbox_preds.view(-1, 4)[pos_masks]
                losses["loss_bbox_aux"] = (
                    self.loss_bbox_aux(bbox_preds_raw, bbox_aux_targets) / num_total_samples
                )

            # keypoint visibility
            kpt_vis_preds = flatten_kpt_vis.view(-1, self.num_keypoints)[pos_masks]
            losses["loss_vis"] = self.loss_vis(kpt_vis_preds, vis_targets, vis_weights)

            # keypoints (regression + coordinate classification)
            kpt_reg_preds = flatten_kpt_decoded.view(-1, self.num_keypoints, 2)[pos_masks]

            if self.loss_mle is not None and getattr(self.loss_mle, "loss_weight", 0) > 0:
                pose_vecs_pos = flatten_pose_vecs.view(-1, flatten_pose_vecs.size(-1))[pos_masks]
                bbox_cs = torch.cat(_bbox_xyxy2cs(bbox_preds_pos, self.bbox_padding), dim=1)
                kpt_cc_preds, pred_hms, sigmas = self.dcc.forward_train(
                    pose_vecs_pos, bbox_cs, pos_priors[..., :2]
                )
                target_hms = self.dcc.generate_target_heatmap(kpt_targets, bbox_cs, sigmas, pos_areas)
                losses["loss_mle"] = self.loss_mle(pred_hms, target_hms, vis_targets)

            if self.proxy_target_cc:
                # use the coordinate-classification prediction as the regression target
                with torch.no_grad():
                    diff_cc = torch.norm(kpt_cc_preds - kpt_targets, dim=-1)
                    diff_reg = torch.norm(kpt_reg_preds - kpt_targets, dim=-1)
                    mask = (diff_reg > diff_cc).float()
                    kpt_weights_reg = vis_targets * mask
                    oks = self.assigner.oks_calculator(kpt_cc_preds, kpt_targets, vis_targets, pos_areas)
                    cls_targets = oks.unsqueeze(1)

                losses["loss_oks"] = self.loss_oks(
                    kpt_reg_preds, kpt_cc_preds.detach(), kpt_weights_reg, pos_areas
                )
            else:
                losses["loss_oks"] = self.loss_oks(kpt_reg_preds, kpt_targets, vis_targets, pos_areas)

            # positive grids get the OKS score as their classification target
            extra_info["overlaps"] = cls_targets
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            obj_targets[pos_masks] = cls_targets.to(obj_targets)

        losses["loss_cls"] = self.loss_cls(cls_preds_all, obj_targets, obj_weights) / num_total_samples
        losses.update(extra_info)
        return losses

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _ensure_priors(self, featmap_sizes, device, dtype):
        if self.mlvl_priors is None or self.featmap_sizes != list(featmap_sizes):
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=dtype, device=device, with_stride=False
            )
            self.flatten_priors = torch.cat(self.mlvl_priors)
            flatten_stride = []
            for priors, stride in zip(self.mlvl_priors, self.featmap_strides):
                flatten_stride.append(priors.new_full((priors.size(0),), stride))
            self.flatten_stride = torch.cat(flatten_stride)
            self.featmap_sizes = list(featmap_sizes)

    @torch.no_grad()
    def predict(self, feats, batch_data_samples, test_cfg=None):
        """Decode a batch of feature maps into per-image pose results.

        Returns one dict per image: ``{"bboxes", "scores", "labels", "keypoints",
        "keypoint_scores", "keypoints_visible"}``.
        """
        cfg = dict(test_cfg or {})
        cls_scores, bbox_preds, _, kpt_vis, pose_vecs = self.forward(feats)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        self._ensure_priors(featmap_sizes, cls_scores[0].device, cls_scores[0].dtype)
        flatten_priors = self.flatten_priors
        flatten_stride = self.flatten_stride

        flatten_cls_scores = self._flatten_predictions(cls_scores).sigmoid()
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
        flatten_pose_vecs = self._flatten_predictions(pose_vecs)
        flatten_bbox_decoded = self.decode_bbox(flatten_bbox_preds, flatten_priors, flatten_stride)

        batch_size = len(batch_data_samples)
        results_list = []

        for i in range(batch_size):
            score_thr = cfg.get("score_thr", 0.01)
            nms_pre = cfg.get("nms_pre", 100000)
            nms_thr = cfg.get("nms_thr", 1.0)

            scores = flatten_cls_scores[i]
            bboxes = flatten_bbox_decoded[i]
            kpt_vis_i = flatten_kpt_vis[i]
            priors_i = flatten_priors
            stride_i = flatten_stride

            scores, labels = scores.max(1, keepdim=True)
            scores, labels, keep_idxs, results = _filter_scores_and_topk(
                scores, score_thr, nms_pre, results=dict(labels=labels[:, 0])
            )
            labels = results["labels"]
            bboxes = bboxes[keep_idxs]
            kpt_vis_i = kpt_vis_i[keep_idxs]
            priors_i = priors_i[keep_idxs]
            stride_i = stride_i[keep_idxs]

            # an explicit slice keeps the shape right when NMS is skipped
            keep_idxs_nms = slice(None)
            if bboxes.numel() > 0 and nms_thr < 1.0:
                keep_idxs_nms = _nms_torch(bboxes, scores, nms_thr)
                bboxes = bboxes[keep_idxs_nms]
                scores = scores[keep_idxs_nms]
                labels = labels[keep_idxs_nms]
                kpt_vis_i = kpt_vis_i[keep_idxs_nms]
                priors_i = priors_i[keep_idxs_nms]
                stride_i = stride_i[keep_idxs_nms]

            if bboxes.numel() > 0:
                pose_vecs_i = flatten_pose_vecs[i][keep_idxs][keep_idxs_nms]
                bbox_cs = torch.cat(_bbox_xyxy2cs(bboxes, self.bbox_padding), dim=1)
                keypoints = self.dcc.forward_test(pose_vecs_i, bbox_cs, priors_i)
            else:
                keypoints = bboxes.new_zeros((0, self.num_keypoints, 2))

            input_size = batch_data_samples[i].get("metainfo", {}).get("input_size", None)
            if input_size is not None and bboxes.numel() > 0:
                bboxes[:, 0::2].clamp_(0, input_size[0])
                bboxes[:, 1::2].clamp_(0, input_size[1])

            results_list.append(
                dict(
                    bboxes=bboxes,
                    scores=scores,
                    labels=labels,
                    keypoints=keypoints,
                    keypoint_scores=keypoints.new_zeros((keypoints.size(0), self.num_keypoints)),
                    keypoints_visible=kpt_vis_i,
                )
            )

        return results_list

    def switch_to_deploy(self, test_cfg=None):
        """Precompute and freeze the priors for deployment exports."""
        if getattr(self, "deploy", False):
            return

        cfg = dict(test_cfg or {})
        input_size = cfg.get("input_size", (640, 640))
        featmap_sizes = []
        for stride in self.featmap_strides:
            featmap_sizes.append((input_size[0] // stride, input_size[1] // stride))
        self._ensure_priors(featmap_sizes, torch.device("cpu"), torch.float32)
        if hasattr(self, "dcc"):
            self.dcc.switch_to_deploy(test_cfg)
        self.deploy = True


__all__ = ["YOLOPoseHead", "SimCCHead", "RTMOHeadModule", "DCC", "RTMOHead"]
