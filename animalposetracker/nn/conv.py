#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: conv.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Convolution family of the AnimalPoseTracker network library:
      - base convolutions : Conv / DWConv / ConvTranspose / DWConvTranspose2d
      - padding helper    : autopad
      - BN folding        : fuse_conv_and_bn / fuse_deconv_and_bn

    IMPORTANT: class names and submodule names (self.conv / self.bn / self.act /
    self.conv_transpose ...) match the training side exactly so trained .pt
    checkpoints load directly via state_dict. Renaming any submodule breaks
    weight compatibility.

Notes:
    - Depends on torch only; no ultralytics / mmcv / mmengine.

Revision History:
    - [2026/9/16] wux024: Initial file creation
    - [2026/10/15] wux024: Split out of modules.py
"""

import math

import torch
import torch.nn as nn


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


def fuse_conv_and_bn(conv, bn):
    """Fold a BatchNorm2d into the Conv2d that feeds it.

    Ports ultralytics' `fuse_conv_and_bn`; the arithmetic itself is PyTorch's own
    `torch.nn.utils.fusion.fuse_conv_bn_weights`, which is also what upstream calls.
    The convolution output is unchanged up to float rounding, so an exported graph gets
    the same numbers with no BatchNorm nodes left -- which matters for inference backends
    that do not fold BN themselves (ncnn / rknn / tflite).

    Args:
        conv (nn.Conv2d): Convolution whose output feeds ``bn`` directly.
        bn (nn.BatchNorm2d): BatchNorm applied immediately after that convolution.

    Returns:
        (nn.Conv2d): The same module, now holding the folded weights, gradients off.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_weights

    if isinstance(bn, nn.Identity):
        return conv.requires_grad_(False)
    conv.weight, conv.bias = fuse_conv_bn_weights(
        conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )
    return conv.requires_grad_(False)


def fuse_deconv_and_bn(deconv, bn):
    """ConvTranspose2d counterpart of :func:`fuse_conv_and_bn`.

    BatchNorm normalizes output channels, but a ConvTranspose2d weight is laid out as
    ``[in, out // groups, kH, kW]``, so the weight is viewed in Conv2d's
    ``[out, in // groups, kH, kW]`` layout for the fold and restored afterwards.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_weights

    if isinstance(bn, nn.Identity):
        return deconv.requires_grad_(False)
    g, (ci, co, *k) = deconv.groups, deconv.weight.shape
    weight = deconv.weight.view(g, ci // g, co, *k).transpose(1, 2).reshape(g * co, ci // g, *k)
    weight, deconv.bias = fuse_conv_bn_weights(
        weight, deconv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )
    deconv.weight = nn.Parameter(weight.view(g, co, ci // g, *k).transpose(1, 2).reshape(ci, co, *k))
    return deconv.requires_grad_(False)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution: one group per input channel pair.

    Groups is set to gcd(c1, c2) so that the convolution is as depth-wise as the channel
    counts allow, without requiring c1 == c2.
    """

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


__all__ = [
    "autopad",
    "Conv",
    "DWConv",
    "ConvTranspose",
    "DWConvTranspose2d",
    "fuse_conv_and_bn",
    "fuse_deconv_and_bn",
]
