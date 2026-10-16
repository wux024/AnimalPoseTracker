#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: block.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Building-block family of the AnimalPoseTracker network library:
      - structural helpers : Concat / Index
      - pooled features    : SPPF
      - regression helper  : DFL (Distribution Focal Loss integral)
      - custom modules     : STEM / CSPNeXtBottleneck / CSPNeXtBlock
      - flow model         : RealNVP (training-only, used by the RLE pose head)

    IMPORTANT: class names and submodule names (self.cv1 / self.cv2 / self.stem /
    self.blocks ...) match the training side exactly so trained .pt checkpoints
    load directly via state_dict. Renaming any submodule breaks weight compatibility.

Notes:
    - Depends on torch only; no ultralytics / mmcv / mmengine.
    - Channel rules for the custom modules are declared in registry.py and must stay
      in sync here.

Revision History:
    - [2026/9/16] wux024: Initial file creation
    - [2026/10/15] wux024: Split out of modules.py
"""

import math

import torch
import torch.nn as nn

from .attention import Attention, ChannelAttention
from .conv import Conv, DWConv


class Concat(nn.Module):
    """Concatenate multiple inputs along the given dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns one particular entry of an input list.

    A structural helper rather than a convolution: it lets a yaml layer pick a single branch
    out of a multi-input list, which is how a single-scale head gets fed from a multi-scale
    backbone in a graph that otherwise only speaks of feature maps.
    """

    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            With the default k=5 and n=3, this module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations and concatenate the feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class DFL(nn.Module):
    """Distribution Focal Loss integral layer: integrates reg_max discrete bins into an expectation.

    The conv kernel is initialized as a buffer and frozen, matching the training side
    (1x1 conv, reg_max -> 1).
    """

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# --------------------------------------------------------------------------------------------
# Custom modules (from the AnimalRTPose)
# Channel rules must stay in sync with their entries in registry.py
# --------------------------------------------------------------------------------------------


class STEM(nn.Module):
    """Spatial-Channel Excitation module.

    Args:
        c1 (int): Number of input channels
        c2 (int): Number of output channels
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.stem = nn.Sequential(
            Conv(c1, c2 // 2, 3, 2, 1), Conv(c2 // 2, c2 // 2, 3, 1, 1), Conv(c2 // 2, c2, 3, 1, 1)
        )

    def forward(self, x):
        return self.stem(x)


class CSPNeXtBottleneck(nn.Module):
    """CSPNeXt bottleneck.

    Args:
        c1, c2 (int): Input / output channels
        e (float): Expansion ratio for the hidden layer
        add_identity (bool): Add a residual connection (requires c1 == c2)
        use_depthwise (bool): Use a depthwise separable convolution
        kernel_size (int): Kernel size of the second convolution
    """

    def __init__(self, c1, c2, e=0.5, add_identity=True, use_depthwise=False, kernel_size=5):
        super().__init__()
        self.use_depthwise = use_depthwise
        conv = DWConv if use_depthwise else Conv
        h = int(c2 * e)
        self.cv1 = conv(c1, h, 3, 1, 1)
        self.cv2 = DWConv(h, c2, kernel_size, 1, kernel_size // 2)
        self.add_identity = add_identity and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add_identity else self.cv2(self.cv1(x))


class CSPNeXtBlock(nn.Module):
    """CSPNeXt block.

    Structure: short_conv / main_conv branches -> n CSPNeXt bottlenecks -> concat
    -> (optional channel attention) -> final_conv

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of stacked bottlenecks
        add_identity (bool): Residual connection inside the bottlenecks
        use_spp (bool): Prepend an SPPF
        use_channelattention (bool): Add channel attention
        use_depthwise (bool): Use depthwise separable convolutions in the bottlenecks
        e (float): Expansion ratio for the hidden layer
    """

    def __init__(
        self, c1, c2, n, add_identity=False, use_spp=False, use_channelattention=False, use_depthwise=False, e=0.5
    ):
        super().__init__()
        self.add_identity = add_identity
        self.use_spp = use_spp
        self.use_depthwise = use_depthwise
        self.use_channelattention = use_channelattention
        m = int(c2 * e)
        if use_spp:
            self.spp = SPPF(c1, c2)
            self.main_conv = Conv(c2, m, 1)
            self.short_conv = Conv(c2, m, 1)
        else:
            self.main_conv = Conv(c1, m, 1)
            self.short_conv = Conv(c1, m, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBottleneck(m, m, 1.0, add_identity, use_depthwise) for _ in range(n)])
        if use_channelattention:
            self.attention = ChannelAttention(m * 2)
        self.final_conv = Conv(m * 2, c2, 1)

    def forward(self, x):
        if self.use_spp:
            x = self.spp(x)
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat([x_main, x_short], dim=1)
        if self.use_channelattention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    Used by the RLE (Residual Log-likelihood Estimation) pose head to model the residual
    distribution of keypoint predictions. Training-only: the flow model is discarded at
    inference, so it adds no inference cost.

    References:
        https://arxiv.org/abs/1605.08803
        Li et al., "Human Pose Regression with Residual Log-likelihood Estimation", ICCV 2021
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    def __init__(self):
        super().__init__()
        # loc/cov are no longer read (the prior is the closed-form standard normal in log_prob)
        # but stay registered so checkpoints saved by older versions still resume.
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Map from the data space to the latent space and compute the log determinant of the Jacobian."""
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Compute the log probability of a given sample in data space."""
        z, log_det = self.backward_p(x)
        # Closed-form log N(z; 0, I) in 2-D; fp32 keeps z**2 from overflowing under AMP.
        return -0.5 * (z.float() ** 2).sum(-1) - math.log(2 * math.pi) + log_det


# --------------------------------------------------------------------------------------------
# YOLO26 backbone blocks (ported for the RTMO variant)
# --------------------------------------------------------------------------------------------


class Bottleneck(nn.Module):
    """Standard bottleneck: two convolutions with an optional residual.

    Ported from the YOLO26 module family for the C2f / C3k2 backbone blocks.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster CSP bottleneck with 2 convolutions (C2f)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 with per-block square kernel sizes (used inside C3k2)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """C2f variant whose inner blocks are Bottleneck, C3k, or Bottleneck + PSABlock."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class PSABlock(nn.Module):
    """Position-Sensitive Attention block: attention and FFN, both with residuals."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2 block with stacked PSABlocks on the main branch."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


__all__ = [
    "Concat",
    "Index",
    "SPPF",
    "DFL",
    "STEM",
    "CSPNeXtBottleneck",
    "CSPNeXtBlock",
    "RealNVP",
    "Bottleneck",
    "C2f",
    "C3",
    "C3k",
    "C3k2",
    "PSABlock",
    "C2PSA",
]
