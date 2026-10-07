#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: modules.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    AnimalPoseTracker network module library.

    Provides every nn.Module needed to build pose estimation networks:
      - generic base layers: Conv / DWConv / DWConvTranspose2d / ConvTranspose / Concat / SPPF / DFL
      - custom modules     : STEM / CSPNeXtBottleneck / CSPNeXtBlock / SPIUpResolution
      - structural helper  : Index
      - ChannelAttention now lives in attention.py (all attention in one place) and is
        re-exported from here so existing imports keep working

    IMPORTANT: class names and submodule names (self.conv / self.bn / self.act / self.cv1 ...)
    match the training side exactly so trained .pt checkpoints load directly via state_dict.
    Renaming any submodule breaks weight compatibility.

Notes:
    - Depends on torch only; no ultralytics / mmcv / mmengine.
    - Channel rules for the custom modules are declared in registry.py and must stay in sync here.

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

import math

import torch
import torch.nn as nn

from .attention import ChannelAttention


def autopad(k, p=None, d=1):
    """Pad to the same output shape."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto pad
    return p


class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm2d + activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depthwise separable convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class ConvTranspose(nn.Module):
    """Transposed convolution: ConvTranspose2d + BatchNorm2d + activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        return self.act(self.conv_transpose(x))


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
    """Spatial Pyramid Pooling - Fast, equivalent to SPP(k=(5, 9, 13))."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution: one group per input channel pair.

    Groups is set to gcd(c1, c2) so that the convolution is as depth-wise as the channel
    counts allow, without requiring c1 == c2.
    """

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


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
# Custom modules (from the AnimalRTPose / SPIPose papers)
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


class SPIUpResolution(nn.Module):
    """SPIPose upsampling module: two transposed convolutions upsampling by 4.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.deconv1 = ConvTranspose(c1, 64, k=4, s=2, p=1)
        self.conv1 = Conv(64, 64, k=3, s=1, p=1)
        self.deconv2 = ConvTranspose(64, c2, k=4, s=2, p=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv1(x)
        x = self.deconv2(x)
        return x


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


__all__ = [
    "autopad",
    "Conv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Concat",
    "Index",
    "SPPF",
    "ChannelAttention",  # re-exported from attention.py
    "DFL",
    "RealNVP",
    "STEM",
    "CSPNeXtBottleneck",
    "CSPNeXtBlock",
    "SPIUpResolution",
]
