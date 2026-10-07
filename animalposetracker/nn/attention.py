#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: attention.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Basic attention blocks, all in one place.

    Inclusion rule: only generic attention components that depend on nothing beyond torch.
    Anything that also needs a *block* (C2f / Bottleneck / C3k) belongs in modules.py, not
    here -- putting it here would create an import cycle.

        ChannelAttention   what to attend to, across channels
        SpatialAttention   where to attend to, in space
        CBAM               ChannelAttention followed by SpatialAttention
        Attention          multi-head self attention in convolutional clothing

    Dependency direction:
        attention.py -> torch only
        modules.py   -> attention.py   (re-exports ChannelAttention for backward compatibility)

    `Attention` needs `Conv`, which lives in modules.py. Importing it at module level would
    make the two files import each other, so it is imported inside __init__ instead.

Notes:
    - Requires torch only.
    - Sub-module names are unchanged from the versions previously defined in modules.py,
      so existing checkpoints still load.

Revision History:
    - [2026/9/16] wux024: Initial file creation. ChannelAttention moved here from modules.py;
      SpatialAttention / CBAM / Attention joined it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChannelAttention", "SpatialAttention", "CBAM", "Attention"]


class ChannelAttention(nn.Module):
    """Channel attention: global average pooling + 1x1 conv + sigmoid rescaling.

    Answers "which channels matter", by squeezing each channel to a single number, learning
    a per-channel gate from it, and multiplying the input by that gate.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise mean and max + 7x7 conv + sigmoid rescaling.

    Answers "where in the image matters", by summarising the channels into two maps (mean
    and max), convolving them into a single spatial gate, and multiplying the input by it.

    Part of CBAM (ECCV 2018), https://arxiv.org/abs/1807.06521
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(
            self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))
        )


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention, then spatial attention.

    The two gates are applied in sequence and are not learned jointly -- CBAM simply chains
    them, which is enough to refine features along both axes.

    https://arxiv.org/abs/1807.06521
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Kernel size for the spatial attention branch.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class Attention(nn.Module):
    """Multi-head self attention over a feature map, implemented with convolutions.

    qkv and the output projection are 1x1 convolutions rather than linear layers, so the
    spatial layout is preserved throughout and the module drops into a convolutional
    backbone without any reshape at its boundary. `pe` is a depthwise 3x3 convolution used
    as a cheap positional encoding.

    Note the `format` attribute: it is None by default. When an exporter sets it to
    'coreml', the forward switches to torch's fused attention kernel, which CoreML can
    lower; the arithmetic is the same.
    """

    format = None

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for the key dimension.
        """
        super().__init__()
        # Deferred import: modules.py imports this file, so a module-level import would cycle.
        from .modules import Conv

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor with shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        if self.format == "coreml" and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
            x = x.transpose(-2, -1).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        else:
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        x = self.proj(x)
        return x
