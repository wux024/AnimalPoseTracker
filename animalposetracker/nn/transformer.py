#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: transformer.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Vision Transformer modules, ported from mmpretrain, used by AnimalViTPose.

        PatchEmbed               image to patch embedding
        MultiheadAttention       qkv + scaled dot-product attention with a projection head
        FFN                      feed-forward network (2 layers by default)
        ViTEncoderLayer          pre-norm encoder layer (ln1 / attn / ln2 / ffn)
        VisionTransformer        the full backbone
        DropPath                 stochastic depth (no parameters)
        resize_pos_embed         bicubic interpolation of patch positional embeddings

    Sub-module names are kept identical to mmpretrain, so checkpoints trained with
    mmpretrain load without key remapping. Only the class name ViTEncoderLayer differs,
    and that is only to leave the name TransformerEncoderLayer free: a class name never
    appears in a state_dict key, so renaming it cannot break checkpoint loading.

Notes:
    - Requires torch. No mmcv / mmengine / mmpretrain dependency.
    - Only the out_type='featmap' path of VisionTransformer is implemented; the other
      output modes raise instead of silently doing the wrong thing.

Revision History:
    - [2026/9/16] wux024: Initial file creation
    - [2026/9/16] wux024: Dropped the ultralytics YOLO / RT-DETR family (TransformerEncoderLayer,
      TransformerLayer, TransformerBlock, MLPBlock, MLP, LayerNorm2d) -- all of them were
      unreferenced and the real ViT now supersedes the two ViT-shaped ones.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PatchEmbed",
    "MultiheadAttention",
    "FFN",
    "ViTEncoderLayer",
    "VisionTransformer",
    "DropPath",
    "resize_pos_embed",
]


def resize_pos_embed(pos_embed: torch.Tensor, target_length: int, num_extra_tokens: int) -> torch.Tensor:
    """Bicubic-interpolate patch positional embeddings onto a different grid size.

    Infer leading class/distillation tokens from the source grid length. This also handles
    AnimalViTPose's ``with_cls_token=False`` target: a 197-token MAE embedding is converted
    to 196 patch tokens, then resized onto the target 16x16 grid.

    Args:
        pos_embed (torch.Tensor): Source embeddings with shape (1, old_length, channels)
        target_length (int): Wanted number of tokens, i.e. the length of the target tensor
        num_extra_tokens (int): Number of leading non-patch tokens required by the target.

    Returns:
        (torch.Tensor): Embeddings with shape (1, target_length, channels)
    """
    if pos_embed.shape[1] == target_length:
        return pos_embed

    num_tokens, channels = pos_embed.shape[1], pos_embed.shape[2]
    num_target_patches = target_length - num_extra_tokens
    source_grid = None
    source_extra_tokens = None
    for candidate_extra_tokens in (num_extra_tokens, 0, 1, 2):
        candidate_patches = num_tokens - candidate_extra_tokens
        candidate_grid = int(round(math.sqrt(candidate_patches))) if candidate_patches > 0 else 0
        if candidate_grid * candidate_grid == candidate_patches:
            source_grid = candidate_grid
            source_extra_tokens = candidate_extra_tokens
            break
    if source_grid is None:
        raise ValueError(f"Cannot infer a square source grid from {num_tokens} position tokens")
    old_grid = source_grid
    new_grid = int(round(math.sqrt(num_target_patches)))
    if new_grid * new_grid != num_target_patches:
        raise ValueError(
            f"Cannot resize a non-square positional embedding: "
            f"{old_grid * old_grid} -> {num_target_patches} patches"
        )

    if source_extra_tokens < num_extra_tokens:
        raise ValueError(
            f"Source has {source_extra_tokens} extra tokens, but target requires {num_extra_tokens}"
        )
    extra = pos_embed[:, :num_extra_tokens, :]
    patch = pos_embed[:, source_extra_tokens:, :]
    patch = patch.reshape(1, old_grid, old_grid, channels).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, channels)

    return torch.cat([extra, patch], dim=1)


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch with probability `drop_prob`.

    Carries no parameters, so it never shows up in a state_dict. In eval mode it is a no-op,
    which keeps the inference graph identical to the training one.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            mask = mask.div_(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


class PatchEmbed(nn.Module):
    """Image to patch embedding, implemented with a convolution.

    Sub-module naming follows mmpretrain: the convolution is `projection`, so a pretrained
    key looks like `patch_embed.projection.weight`.

    Args:
        in_channels (int): Number of input channels.
        embed_dims (int): Embedding dimension.
        kernel_size (int): Patch size, used as the convolution kernel.
        stride (int): Convolution stride, defaults to the patch size.
        padding (int): Convolution padding.
        dilation (int): Convolution dilation.
        bias (bool): Whether the projection convolution carries a bias.
        input_size (tuple, optional): (H, W) of the input, used to pre-compute the output grid.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int = 16,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if input_size is not None:
            self.init_input_size = tuple(input_size)
            h_out = (input_size[0] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            w_out = (input_size[1] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Split the image into patches and project them.

        Args:
            x (torch.Tensor): Input image with shape (B, C, H, W)

        Returns:
            tuple: (tokens with shape (B, H*W, embed_dims), spatial shape (H, W))
        """
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


class MultiheadAttention(nn.Module):
    """Multi-head self attention with a projection head, mirroring mmpretrain's naming.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        input_dims (int, optional): Input dimension, defaults to embed_dims.
        attn_drop (float): Dropout on the attention map. Kept as a plain float on purpose --
            mmpretrain does the same, so it contributes no state_dict key.
        proj_drop (float): Dropout after the projection.
        qkv_bias (bool): Whether qkv carries a bias.
        proj_bias (bool): Whether the projection carries a bias.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        input_dims: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        # mmpretrain always builds these two, even when they are inert.
        self.out_drop = nn.Dropout(0.0)
        self.gamma1 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input with shape (B, N, C)

        Returns:
            (torch.Tensor): Output with shape (B, N, embed_dims)
        """
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.attn_drop, training=self.training)

        out = (attn @ v).transpose(1, 2).reshape(b, n, self.embed_dims)
        out = self.proj(out)
        return self.out_drop(self.gamma1(self.proj_drop(out)))


class FFN(nn.Module):
    """Feed-forward network mirroring mmcv's FFN, including its sub-module layout.

    With num_fcs=2 the learnable parameters live at `layers.0.0` and `layers.1`, which is
    what the pretrained keys expect.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act: Optional[nn.Module] = None,
        ffn_drop: float = 0.0,
        add_identity: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    nn.GELU() if act is None else act,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.dropout_layer = nn.Identity()
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. The residual connection to `x` is implied, as in mmcv."""
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        return x + self.dropout_layer(out)


class ViTEncoderLayer(nn.Module):
    """A single pre-norm ViT encoder layer: ln1 -> attn -> add, then ln2 -> ffn -> add.

    Sub-modules keep the mmpretrain names (ln1 / attn / ln2 / ffn), so checkpoints still load.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        num_fcs: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dims, eps=eps)
        self.attn = MultiheadAttention(
            embed_dims,
            num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
        )
        self.ln2 = nn.LayerNorm(embed_dims, eps=eps)
        self.ffn = FFN(
            embed_dims,
            feedforward_channels,
            num_fcs=num_fcs,
            act=nn.GELU(),
            ffn_drop=drop_rate,
            add_identity=False,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input with shape (B, N, C)

        Returns:
            (torch.Tensor): Output with shape (B, N, C)
        """
        x = x + self.drop_path(self.attn(self.ln1(x)))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone, mmpretrain-compatible.

    Only the `out_type='featmap'` path is implemented, because that is the only one
    AnimalViTPose uses: the encoder output is reshaped back to a feature map (B, C, H, W)
    so the pose head can consume it like any other backbone output.

    `with_cls_token=False` matches the AnimalViTPose configs: no class token is created and
    `num_extra_tokens` is 0. Loading a checkpoint that does contain a class token therefore
    needs `strict=False`; the token is simply dropped.

    Args:
        arch (str | dict): Architecture preset from ARCH_ZOO, or an explicit dict.
        img_size (int | tuple): Input resolution.
        patch_size (int): Patch size.
        in_channels (int): Number of input channels.
        out_indices (int): Which layer to take the output from, -1 for the last one.
        drop_rate (float): Dropout after the positional embedding and inside the FFN.
        drop_path_rate (float): Maximum stochastic-depth rate, linearly scaled across layers.
        qkv_bias (bool): Whether qkv carries a bias.
        final_norm (bool): Whether to apply a LayerNorm after the encoder.
        out_type (str): Output form; only 'featmap' is implemented.
        with_cls_token (bool): Whether to prepend a class token.
        patch_cfg (dict, optional): Extra arguments forwarded to PatchEmbed.
    """

    ARCH_ZOO = {
        "small": dict(embed_dims=384, num_layers=12, num_heads=12, feedforward_channels=1536),
        "base": dict(embed_dims=768, num_layers=12, num_heads=12, feedforward_channels=3072),
        "large": dict(embed_dims=1024, num_layers=24, num_heads=16, feedforward_channels=4096),
        "huge": dict(embed_dims=1280, num_layers=32, num_heads=16, feedforward_channels=5120),
    }

    OUT_TYPES = {"featmap"}

    def __init__(
        self,
        arch: Union[str, dict] = "base",
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_indices: int = -1,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        final_norm: bool = True,
        out_type: str = "featmap",
        with_cls_token: bool = False,
        patch_cfg: Optional[dict] = None,
    ):
        super().__init__()

        if out_type not in self.OUT_TYPES:
            raise NotImplementedError(
                f"out_type '{out_type}' is not implemented, only {sorted(self.OUT_TYPES)}. "
                f"Add it here if you really need it."
            )
        self.out_type = out_type
        self.with_cls_token = with_cls_token

        if isinstance(arch, str):
            if arch not in self.ARCH_ZOO:
                raise KeyError(f"Unknown arch '{arch}'. Available: {sorted(self.ARCH_ZOO)}")
            arch = self.ARCH_ZOO[arch]
        elif isinstance(arch, dict):
            arch = arch.copy()
        else:
            raise TypeError(f"arch must be a str or a dict, got {type(arch)}")

        # Normalize img_size / patch_size to (h, w)
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        kernel = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        if any(s % p != 0 for s, p in zip(img_size, kernel)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")

        embed_dims = arch["embed_dims"]
        num_layers = arch["num_layers"]
        num_heads = arch["num_heads"]
        feedforward_channels = arch["feedforward_channels"]
        if num_heads <= 0 or embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        patch_kwargs = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        patch_kwargs.update(patch_cfg or {})
        self.patch_embed = PatchEmbed(**patch_kwargs)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        self.num_extra_tokens = 1 if with_cls_token else 0
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        elif isinstance(out_indices, (list, tuple)):
            out_indices = list(out_indices)
        else:
            raise TypeError(f"out_indices must be int or list, got {type(out_indices)}")
        for i in out_indices:
            if i < -num_layers or i >= num_layers:
                raise ValueError(f"out_indices {out_indices} out of range for {num_layers} layers")
        self.out_indices = [i if i >= 0 else num_layers + i for i in out_indices]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList(
            ViTEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                drop_rate=drop_rate,
                attn_drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
            )
            for i in range(num_layers)
        )

        self.final_norm = final_norm
        self.ln1 = nn.LayerNorm(embed_dims, eps=1e-6) if final_norm else nn.Identity()

        self.init_weights()
        # A pretrained positional embedding usually sits on a different grid; interpolate it on load.
        self.register_load_state_dict_pre_hook(self._prepare_pos_embed)

    def init_weights(self) -> None:
        """Initialize the positional embedding, the class token and the model layers."""
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def _prepare_pos_embed(self, *args, **kwargs) -> None:
        """Interpolate a pretrained positional embedding onto this model's grid before loading."""
        # PyTorch's register_load_state_dict_pre_hook may pass the owning module explicitly;
        # accept both documented callback layouts while keeping this as a bound method.
        if len(args) >= 3 and isinstance(args[0], nn.Module):
            _module, state_dict, prefix = args[:3]
        elif len(args) >= 2:
            state_dict, prefix = args[:2]
        else:
            raise TypeError("Unexpected state-dict pre-hook signature")
        key = prefix + "pos_embed"
        if key not in state_dict or state_dict[key].shape == self.pos_embed.shape:
            return
        state_dict[key] = resize_pos_embed(state_dict[key], self.pos_embed.shape[1], self.num_extra_tokens)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input image with shape (B, C, H, W)

        Returns:
            (torch.Tensor): Feature map with shape (B, embed_dims, H/patch, W/patch)
        """
        b = x.shape[0]
        x, patch_resolution = self.patch_embed(x)  # (B, N, C)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)

        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)
            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return outs[0] if len(self.out_indices) == 1 else tuple(outs)

    def _format_output(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """Strip the extra tokens and reshape the token sequence back to a feature map."""
        if self.num_extra_tokens:
            x = x[:, self.num_extra_tokens:, :]
        h, w = hw
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], h, w).contiguous()
        return x
