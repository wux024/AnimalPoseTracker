#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: head.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.1

Overview:
    Two heads live here. They are alternatives, not siblings -- a model picks one:

        1. YOLOPoseHead  anchor-based regression head, used by AnimalRTPose / SPIPose
        2. SimCCHead     1D coordinate classification head, used by AnimalViTPose

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

from .modules import Conv, DFL, RealNVP
from .tal import dist2bbox, make_anchors


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
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid(), self.kpts_decode(bs, kpt)), 1)

    def bias_init(self):
        """Initialize biases to suppress the high-confidence prediction storm early in training."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
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


__all__ = ["YOLOPoseHead", "SimCCHead"]
