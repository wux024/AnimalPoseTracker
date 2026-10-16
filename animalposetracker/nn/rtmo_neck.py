#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: rtmo_neck.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/10/16
Version: 1.0

Overview:
    Neck family of the RTMO model (ported from mmpose / mmdet, pure torch).

        RepVGGBlock             3x3 + 1x1 (+ optional identity) block with BN folding
        CSPRepLayer             CSP branch pair of RepVGGBlocks
        DetrSelfAttention       DETR-style self attention (query/key positional encoding
                                added, residual applied internally)
        DetrTransformerEncoderLayer / DetrTransformerEncoder
                                pre-norm-free encoder stack used by the HybridEncoder
        ChannelMapper           per-scale 1x1 style channel projection
        HybridEncoder           RT-DETR hybrid encoder: transformer encoder + FPN + PAN

    Sub-module names (`branch_3x3` / `branch_1x1` / `self_attn` / `ffn` / `norms` ...)
    are kept identical to the mmpose / mmcv originals, so state_dict keys line up with
    reference checkpoints.

Notes:
    - Requires torch only. mmcv's ConvModule is replaced by the local Conv (whose
      `conv` / `bn` sub-module layout matches); mmcv's FFN is replaced by the local
      FFN (`nn.transformer`), which mirrors mmcv's layout including the residual.
    - The mmcv MultiheadAttention is reimplemented as DetrSelfAttention with the same
      semantics: query/key positional encodings are added before attention and the
      residual (identity) is applied inside the module.

Revision History:
    - [2026/10/16] wux024: Initial file creation (RTMO port, batch 3)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .conv import Conv
from .transformer import FFN, SinePositionalEncoding

__all__ = [
    "RepVGGBlock",
    "CSPRepLayer",
    "DetrSelfAttention",
    "DetrTransformerEncoderLayer",
    "DetrTransformerEncoder",
    "ChannelMapper",
    "HybridEncoder",
]


class RepVGGBlock(nn.Module):
    """RepVGG block: 3x3 + 1x1 convolutions plus an optional normalized identity.

    Ported from mmpose / mmdet. `get_equivalent_kernel_bias` folds the three branches
    into a single convolution for deployment via `switch_to_deploy`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        act: Optional[nn.Module] = None,
        without_branch_norm: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # A normalized identity branch only makes sense when shapes line up; the
        # RTMO config keeps `without_branch_norm=True`, i.e. no identity norm.
        self.branch_norm = None
        if out_channels == in_channels and stride == 1 and padding == dilation and not without_branch_norm:
            self.branch_norm = nn.BatchNorm2d(in_channels)

        self.branch_3x3 = Conv(
            self.in_channels,
            self.out_channels,
            3,
            s=self.stride,
            p=self.padding,
            g=self.groups,
            d=self.dilation,
            act=False,
        )
        self.branch_1x1 = Conv(self.in_channels, self.out_channels, 1, g=self.groups, act=False)

        self.act = act if act is not None else nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.branch_norm is None:
            branch_norm_out = 0
        else:
            branch_norm_out = self.branch_norm(x)

        out = self.branch_3x3(x) + self.branch_1x1(x) + branch_norm_out
        return self.act(out)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad a 1x1 kernel to 3x3 with zeros."""
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Derive the equivalent kernel and bias of one branch."""
        if branch is None:
            return 0, 0

        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=torch.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        """Fold the conv-bn branches into one kernel and bias (differentiable)."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.branch_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernelid, biasid = (0, 0) if self.branch_norm is None else self._fuse_bn_tensor(self.branch_norm)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self, test_cfg: Optional[Dict] = None):
        """Replace the branches with the single folded convolution (inference only)."""
        if getattr(self, "deploy", False):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True,
        )
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "branch_norm"):
            self.__delattr__("branch_norm")

        def _forward(self, x):
            return self.act(self.conv_reparam(x))

        self.forward = _forward.__get__(self)
        self.deploy = True


class CSPRepLayer(nn.Module):
    """CSP branch pair of RepVGG blocks, followed by a fused 1x1 projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        widen_factor: float = 1.0,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        hidden_channels = int(out_channels * widen_factor)
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1)
        self.bottlenecks = nn.Sequential(
            *(RepVGGBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks))
        )
        if hidden_channels != out_channels:
            self.conv3 = Conv(hidden_channels, out_channels, 1)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_1 = self.bottlenecks(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class DetrSelfAttention(nn.Module):
    """DETR-style self attention with positional encodings and an internal residual.

    Reimplements mmcv's MultiheadAttention semantics used by the RT-DETR encoder:

    - `query_pos` / `key_pos` are added onto the query / key before projection;
    - the value projection shares the input with the query;
    - the residual is applied inside: ``out = (query + query_pos) + out_proj(attn)``.

    Uses PyTorch's `nn.MultiheadAttention` for the projection + attention math with
    `batch_first=True`, then re-applies the mmcv residual convention.
    """

    def __init__(self, embed_dims: int = 256, num_heads: int = 8, dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        if not batch_first:
            raise ValueError("DetrSelfAttention requires batch_first=True")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if key_pos is None:
            if query_pos is not None:
                key_pos = query_pos

        identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)
        return identity + out


class DetrTransformerEncoderLayer(nn.Module):
    """Single DETR encoder layer: self attention -> norm -> FFN -> norm.

    Mirrors mmpose's layout (`self_attn` / `ffn` / `norms`), with the local FFN that
    already carries the residual internally.
    """

    def __init__(
        self,
        self_attn_cfg: Optional[Dict] = None,
        ffn_cfg: Optional[Dict] = None,
        embed_dims: int = 256,
    ):
        super().__init__()
        self_attn_cfg = dict(self_attn_cfg or {})
        ffn_cfg = dict(ffn_cfg or {})

        self.self_attn = DetrSelfAttention(
            embed_dims=self_attn_cfg.get("embed_dims", embed_dims),
            num_heads=self_attn_cfg.get("num_heads", 8),
            dropout=self_attn_cfg.get("dropout", 0.0),
        )
        self.embed_dims = self.self_attn.embed_dims

        self.ffn = FFN(
            embed_dims=ffn_cfg.get("embed_dims", self.embed_dims),
            feedforward_channels=ffn_cfg.get("feedforward_channels", self.embed_dims * 4),
            num_fcs=ffn_cfg.get("num_fcs", 2),
            act=ffn_cfg.get("act", nn.GELU()),
            ffn_drop=ffn_cfg.get("ffn_drop", 0.0),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dims) for _ in range(2)])

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)
        return query


class DetrTransformerEncoder(nn.Module):
    """Stack of DETR encoder layers (no checkpointing — not needed here)."""

    def __init__(self, num_layers: int, layer_cfg: Optional[Dict] = None, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DetrTransformerEncoderLayer(**(layer_cfg or {})) for _ in range(num_layers)])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, **kwargs) -> Tensor:
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


class ChannelMapper(nn.Module):
    """Project each input scale to `out_channels` with a single convolution."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv(in_channel, out_channels, kernel_size, p=(kernel_size - 1) // 2, act=act)
                for in_channel in in_channels
            ]
        )

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, ...]:
        assert len(inputs) == len(self.convs)
        return tuple(conv(x) for conv, x in zip(self.convs, inputs))


class HybridEncoder(nn.Module):
    """RT-DETR hybrid encoder neck: transformer encoder + FPN + PAN.

    Ported from mmpose with the mmcv / registry pieces replaced: ConvModule -> local
    Conv, DetrTransformerEncoder -> the local port above, and the projector passed in
    as an instantiated module instead of a `MODELS.build` config.

    Args:
        in_channels (List[int]): Channels of the input feature maps.
        feat_strides (List[int]): Strides of the input feature maps.
        hidden_dim (int): Hidden dimension of the projections.
        use_encoder_idx (List[int]): Which feature maps go through the transformer.
        num_encoder_layers (int): Layers per transformer encoder.
        encoder_cfg (dict): Config for the transformer encoder layers, built as
            ``dict(self_attn_cfg=..., ffn_cfg=...)``.
        output_indices (List[int], optional): Output scale selection.
        projector: An optional instantiated module applied to the outputs.
        pe_temperature (int): Positional encoding temperature.
        spe_learnable (bool): Whether the positional encoding base is learnable.
        widen_factor / deepen_factor (float): CSPRepLayer depth / width multipliers.
    """

    def __init__(
        self,
        in_channels: List[int] = (512, 1024, 2048),
        feat_strides: List[int] = (8, 16, 32),
        hidden_dim: int = 256,
        use_encoder_idx: List[int] = (2,),
        num_encoder_layers: int = 1,
        encoder_cfg: Optional[Dict] = None,
        output_indices: Optional[List[int]] = None,
        projector=None,
        pe_temperature: int = 10000,
        widen_factor: float = 1.0,
        deepen_factor: float = 1.0,
        spe_learnable: bool = False,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.output_indices = output_indices

        # channel projection
        self.input_proj = nn.ModuleList(
            [Conv(in_channel, hidden_dim, 1, act=False) for in_channel in self.in_channels]
        )

        # transformer encoder
        if len(self.use_encoder_idx) > 0:
            self.encoder = nn.ModuleList(
                [
                    DetrTransformerEncoder(num_encoder_layers, encoder_cfg)
                    for _ in range(len(self.use_encoder_idx))
                ]
            )

        pos_enc_dim = self.hidden_dim // 2
        self.sincos_pos_enc = SinePositionalEncoding(
            pos_enc_dim,
            learnable=spe_learnable,
            temperature=self.pe_temperature,
            spatial_dim=2,
        )

        # top-down FPN
        lateral_convs = []
        fpn_blocks = []
        for _idx in range(len(self.in_channels) - 1, 0, -1):
            lateral_convs.append(Conv(hidden_dim, hidden_dim, 1))
            fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * deepen_factor),
                    widen_factor=widen_factor,
                )
            )
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.fpn_blocks = nn.ModuleList(fpn_blocks)

        # bottom-up PAN
        downsample_convs = []
        pan_blocks = []
        for _idx in range(len(self.in_channels) - 1):
            downsample_convs.append(Conv(hidden_dim, hidden_dim, 3, s=2, p=1))
            pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * deepen_factor),
                    widen_factor=widen_factor,
                )
            )
        self.downsample_convs = nn.ModuleList(downsample_convs)
        self.pan_blocks = nn.ModuleList(pan_blocks)

        self.projector = projector

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        if len(inputs) != len(self.in_channels):
            raise ValueError("number of inputs must match in_channels")

        proj_feats = [self.input_proj[i](inputs[i]) for i in range(len(inputs))]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1).contiguous()
                pos_enc = self.sincos_pos_enc(size=(h, w))
                pos_enc = pos_enc.transpose(-1, -2).reshape(1, h * w, -1)
                memory = self.encoder[i](src_flatten, query_pos=pos_enc, key_padding_mask=None)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).contiguous().view([-1, self.hidden_dim, h, w])

        # top-down FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        if self.output_indices is not None:
            outs = [outs[i] for i in self.output_indices]

        if self.projector is not None:
            outs = self.projector(outs)

        return tuple(outs)
