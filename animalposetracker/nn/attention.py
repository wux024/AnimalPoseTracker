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

    Inclusion rule: only generic attention components. Anything that also needs a *block*
    (C2f / Bottleneck / C3k) belongs in block.py, not here -- putting it here would create
    an import cycle. GAUEncoder additionally pulls DropPath / ScaleNorm /
    SinePositionalEncoding from transformer.py, which stays torch-only, so there is still
    no cycle.

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import DropPath, ScaleNorm, SinePositionalEncoding

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Attention",
    "ChannelWiseScale",
    "GAUEncoder",
]


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
        from .conv import Conv

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


class ChannelWiseScale(nn.Module):
    """Scale a vector by element-wise multiplication with a learnable vector.

    Used as the residual scaling of the GAU encoder.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float): The initial value of the scale vector. Defaults to 1.0.
        trainable (bool): Whether the scale vector is trainable. Defaults to True.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class GAUEncoder(nn.Module):
    """Gated Attention Unit (GAU) Encoder.

    Ported from mmpose (`mmpose/models/utils/transformer.py`); used by the RTMO
    head's Dynamic Coordinate Classifier. Reference: "Transformer Quality in
    Linear Time" (arXiv:2202.10447).

    Args:
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int): The expansion factor of the intermediate token
            dimension. Defaults to 2.
        s (int): The self-attention feature dimension. Defaults to 128.
        eps (float): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float): The dropout rate. Defaults to 0.0.
        drop_path (float): The drop path rate. Defaults to 0.0.
        act_fn (str): 'SiLU' or 'ReLU'. Defaults to 'SiLU'.
        bias (bool): Whether to use bias in linear layers. Defaults to False.
        pos_enc (str): 'none', 'add' or 'rope'. Defaults to 'none'.
        spatial_dim (int): The spatial dimension of the inputs. Defaults to 1.
    """

    def __init__(
        self,
        in_token_dims,
        out_token_dims,
        expansion_factor=2,
        s=128,
        eps=1e-5,
        dropout_rate=0.0,
        drop_path=0.0,
        act_fn="SiLU",
        bias=False,
        pos_enc: str = "none",
        spatial_dim: int = 1,
    ):
        super().__init__()
        self.s = s
        self.bias = bias
        self.pos_enc = pos_enc
        self.in_token_dims = in_token_dims
        self.spatial_dim = spatial_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        self._build_layers()

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == "SiLU":
            self.act_fn = nn.SiLU(True)
        else:
            self.act_fn = nn.ReLU(True)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = ChannelWiseScale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def _build_layers(self):
        self.uv = nn.Linear(self.in_token_dims, 2 * self.e + self.s, bias=self.bias)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))

    def _forward(self, x, mask=None, pos_enc=None):
        """GAU forward function."""
        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
        # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
        dim = base.ndim - self.gamma.ndim + 1
        gamma = self.gamma.view(*((1,) * dim), *self.gamma.size())
        beta = self.beta.view(*((1,) * dim), *self.beta.size())
        base = base.unsqueeze(-2) * gamma + beta
        # [B, K, 2, s] -> [B, K, s], [B, K, s]
        q, k = torch.unbind(base, dim=-2)

        if self.pos_enc == "rope":
            q = SinePositionalEncoding.apply_rotary_pos_enc(q, pos_enc, self.spatial_dim)
            k = SinePositionalEncoding.apply_rotary_pos_enc(k, pos_enc, self.spatial_dim)
        elif self.pos_enc == "add":
            pos_enc = pos_enc.reshape(*((1,) * (q.ndim - 2)), q.size(-2), q.size(-1))
            q = q + pos_enc
            k = k + pos_enc

        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.matmul(q, k.transpose(-1, -2))

        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if mask is not None:
            kernel = kernel * mask

        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)

        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.matmul(kernel, v)
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x, mask=None, pos_enc=None):
        """Forward function."""
        out = self.drop_path(self._forward(x, mask=mask, pos_enc=pos_enc))
        if self.shortcut:
            return self.res_scale(x) + out
        return out
