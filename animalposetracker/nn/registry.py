#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: registry.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Explicit registry mapping yaml module names to build rules.

    This is the core of the model-building layer. Every module registers one rule here,
    and the rule translates yaml arguments into the arguments actually passed to the
    nn.Module, plus the output channel count and spatial ratio of that layer.

    Difference from ultralytics:
        Upstream parse_model resolves class names through globals() and supports hundreds of
        modules with 19 hard-coded `if m is XXX` branches. This project only supports its own
        small set of modules, so:
          - the module table is an explicit dict instead of polluting globals()
          - each module's logic is a standalone function with no branch nesting
          - adding a module means adding one entry to RULES; no existing code changes

Notes:
    - This module does not import torch. Channel / spatial inference is pure arithmetic and is
      decoupled from instantiation, so a yaml can be validated without torch (see spec.py).

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

__all__ = [
    "BuildContext",
    "Rule",
    "RULES",
    "register",
    "get_rule",
    "make_divisible",
]


def make_divisible(x: float, divisor: int) -> int:
    """Round up to the nearest multiple of divisor (same behavior as ultralytics)."""
    import math

    return math.ceil(x / divisor) * divisor


@dataclass
class BuildContext:
    """Build context: derived from the top-level keys of the model yaml, used by all layers."""

    nc: int = 80  # number of classes
    kpt_shape: Sequence[int] = (17, 3)  # (number of keypoints, dimensions)
    scale: Optional[str] = None  # current scale, e.g. 'm'
    scales: Optional[Dict[str, Sequence[float]]] = None  # {scale: [depth, width, max_channels]}
    depth: float = 1.0
    width: float = 1.0
    max_channels: float = float("inf")
    top_level: Dict[str, Any] = field(default_factory=dict)  # yaml top-level keys, for bare-identifier resolution

    def __post_init__(self):
        # Same as ultralytics: when depth/width are not given explicitly, take the first scale entry
        if self.scales and self.scale is None:
            self.scale = tuple(self.scales.keys())[0]
        if self.scales and self.scale in self.scales:
            self.depth, self.width, self.max_channels = self.scales[self.scale]


# ------------------------------------------------------------------------------------------------
# Uniform signature of a rule function
#   fn(args, n, ch_in, ctx) -> (new_args, new_n, c2, spatial)
#     args    : argument list written in the yaml (already resolved)
#     n       : repeats column from the yaml
#     ch_in   : input channels; int for a single input, list[int] for multi-input (Concat/Pose)
#     ctx     : BuildContext
#   returns
#     new_args: arguments actually passed to the nn.Module
#     new_n   : outer repeat count (modules that move repeats into args return 1)
#     c2      : output channels of this layer
#     spatial : output/input size ratio (1.0 unchanged, 2.0 downsample by 2, 0.5 upsample by 2)
# ------------------------------------------------------------------------------------------------


def _conv(args, n, ch_in, ctx):
    """Standard / depthwise / transposed convolution: channels scaled by width, rounded up to a multiple of 8."""
    c2 = make_divisible(min(args[0], ctx.max_channels) * ctx.width, 8)
    stride = args[1] if len(args) > 1 else 1
    if stride < 1:  # ConvTranspose: stride means upsampling
        spatial = 1.0 / stride
    else:
        spatial = float(stride)
    return [ch_in, c2, *args[1:]], n, c2, spatial


def _stem(args, n, ch_in, ctx):
    """STEM: channels scaled by width directly (no multiple-of-8 alignment), always downsamples by 2."""
    c2 = int(args[0] * ctx.width)
    return [ch_in, c2], n, c2, 2.0


def _cspnext_block(args, n, ch_in, ctx):
    """CSPNeXtBlock: repeats move into the arguments as the internal bottleneck count, outer repeats set to 1."""
    c2 = int(args[0] * ctx.width)
    return [ch_in, c2, n, *args[1:]], 1, c2, 1.0


def _spi_upresolution(args, n, ch_in, ctx):
    """SPIUpResolution: channels are not width-scaled; two transposed convs upsample by 4 in total."""
    return [ch_in, args[0]], n, args[0], 0.25


def _passthrough(args, n, ch_in, ctx):
    """Layers that keep channels and spatial size (channel part of nn.Upsample is handled separately)."""
    return list(args), n, ch_in, 1.0


def _upsample(args, n, ch_in, ctx):
    """nn.Upsample: channels unchanged, upsampled by scale_factor (yaml form [None, 2, "nearest"])."""
    factor = args[1] if len(args) > 1 else args[0]
    spatial = 1.0 / float(factor) if factor else 1.0
    return list(args), n, ch_in, spatial


def _concat(args, n, ch_in, ctx):
    """Concat: output channels are the sum of all inputs."""
    return list(args), n, sum(ch_in), 1.0


def _pose_head(args, n, ch_in, ctx):
    """Anchor-based pose head: insert the multi-scale input channel list right after (nc, kpt_shape).

    Extra yaml arguments (e.g. the optional ``rle`` flag) are passed through after the channel
    list, so the positional order matches ``YOLOPoseHead(nc, kpt_shape, ch, rle=False)``:
        yaml [nc, kpt_shape]        -> YOLOPoseHead(nc, kpt_shape, ch)
        yaml [nc, kpt_shape, True]  -> YOLOPoseHead(nc, kpt_shape, ch, True)   # RLE mode

    On c2: the head outputs a decoded prediction tensor (bs, 4+nc+nk, anchors), not a feature
    map, so "output channels" has no meaning here. Returning ch_in[-1] only keeps the channel
    table well-formed (ultralytics does not rewrite c2 in this branch, which is equivalent to
    carrying the previous layer's value). Because the head must be the last layer, this
    placeholder is never read by a later layer -- spec.parse enforces that.
    """
    new_args = [*args[:2], list(ch_in), *args[2:]]
    return new_args, n, ch_in[-1], 1.0


def _vit(args, n, ch_in, ctx):
    """VisionTransformer backbone.

    yaml form:
        [embed_dims, num_layers, num_heads, feedforward_channels, patch_size, img_size]
    an optional 7th entry is a patch_cfg dict forwarded to PatchEmbed, e.g. {padding: 2}.

    The architecture is spelled out positionally rather than by arch name, because this
    function also runs inside spec.parse, which must stay torch-free and therefore cannot
    read VisionTransformer.ARCH_ZOO.

    Note the spatial ratio: a ViT with patch_size 16 on a 256x256 input returns a 16x16
    feature map, so spatial = patch_size / img_size.
    """
    if len(args) < 6:
        raise ValueError(
            f"ViT needs [embed_dims, num_layers, num_heads, feedforward_channels, "
            f"patch_size, img_size], got {args}"
        )
    embed_dims, num_layers, num_heads, feedforward_channels, patch_size, img_size = args[:6]
    patch_cfg = args[6] if len(args) > 6 else None

    if img_size % patch_size != 0:
        raise ValueError(f"ViT: img_size {img_size} must be divisible by patch_size {patch_size}")

    kwargs = dict(
        arch=dict(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
        ),
        img_size=img_size,
        patch_size=patch_size,
        in_channels=ch_in,
    )
    if patch_cfg:
        kwargs["patch_cfg"] = patch_cfg

    return [kwargs], n, embed_dims, patch_size / img_size


def _simcc_head(args, n, ch_in, ctx):
    """SimCC head: two 1D coordinate classifications per keypoint.

    yaml form:
        [out_channels, input_size, in_featuremap_size]
        plus optional [simcc_split_ratio, deconv_out_channels, deconv_kernel_sizes].

    in_featuremap_size has to be given explicitly. spec.parse tracks spatial *ratios*, not
    absolute sizes, so the height and width of the incoming feature map are not knowable
    here -- and SimCCHead's Linear layers bake those dimensions into their weight shapes.

    On c2: the head emits two 1D distributions, not a feature map, so "output channels"
    has no meaning. Returning the input channel count only keeps the channel table
    well-formed; because the head must be the last layer, nothing reads it.
    """
    if len(args) < 3:
        raise ValueError(
            f"SimCCHead needs at least [out_channels, input_size, in_featuremap_size], got {args}"
        )
    out_channels, input_size, in_featuremap_size = args[:3]
    simcc_split_ratio = args[3] if len(args) > 3 else 2.0
    deconv_out_channels = args[4] if len(args) > 4 else (256,)
    deconv_kernel_sizes = args[5] if len(args) > 5 else (4,)

    in_channels = ch_in[-1] if isinstance(ch_in, (list, tuple)) else ch_in

    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        input_size=tuple(input_size),
        in_featuremap_size=tuple(in_featuremap_size),
        simcc_split_ratio=simcc_split_ratio,
        deconv_out_channels=tuple(deconv_out_channels),
        deconv_kernel_sizes=tuple(deconv_kernel_sizes),
    )
    return [kwargs], n, in_channels, 1.0


@dataclass
class Rule:
    """One module build rule.

    Args:
        name: Module name as written in the yaml
        cls: Class name in modules.py / head.py
        fn: Argument resolution function, see the uniform signature above
        multi_input: Whether the module accepts multiple inputs (from column may be a list)
        repeatable: Whether depth scaling applies (False keeps n at 1)
        spatial: Static spatial ratio; None means the rule function decides
        note: Short description
        is_head: Head layer that outputs decoded predictions; must be the last layer
    """

    name: str
    cls: str
    fn: Callable
    multi_input: bool = False
    repeatable: bool = True
    spatial: Optional[float] = None
    note: str = ""
    is_head: bool = False


RULES: Dict[str, Rule] = {}


def register(rule: Rule) -> Rule:
    """Register a rule; duplicate names raise an error instead of silently overriding."""
    if rule.name in RULES:
        raise KeyError(f"Duplicate module rule registration: {rule.name}")
    RULES[rule.name] = rule
    return rule


def get_rule(name: str) -> Rule:
    """Fetch a rule; unknown names raise an error listing the available modules."""
    if name not in RULES:
        raise KeyError(f"Unregistered module '{name}'. Registered: {sorted(RULES)}")
    return RULES[name]


# ------------------------------------------------------------------------------------------------
# Module list: adding a module only requires one entry here
# ------------------------------------------------------------------------------------------------
register(Rule("Conv", "Conv", _conv, note="Standard convolution, width-scaled and aligned to 8"))
register(Rule("DWConv", "DWConv", _conv, note="Depthwise separable convolution"))
register(Rule("ConvTranspose", "ConvTranspose", _conv, note="Transposed convolution"))
register(Rule("Concat", "Concat", _concat, multi_input=True, note="Multi-input concat, channels summed"))
register(Rule("SPPF", "SPPF", _conv, note="Spatial pyramid pooling (commonly follows STEM)"))
register(Rule("nn.Upsample", "nn.Upsample", _upsample, repeatable=False,
              note="Native torch upsampling, built directly from torch.nn"))
register(Rule("STEM", "STEM", _stem, note="Custom: Spatial-Channel Excitation"))
register(Rule("CSPNeXtBottleneck", "CSPNeXtBottleneck", _passthrough, note="Custom: CSPNeXt bottleneck"))
register(Rule("CSPNeXtBlock", "CSPNeXtBlock", _cspnext_block, note="Custom: repeats move into args"))
register(Rule("SPIUpResolution", "SPIUpResolution", _spi_upresolution, note="Custom: SPIPose upsampling"))
register(Rule("YOLOPoseHead", "YOLOPoseHead", _pose_head, multi_input=True, repeatable=False, is_head=True,
              note="Anchor-based pose head (boxes + classes + keypoints)"))
register(Rule("ViT", "VisionTransformer", _vit, repeatable=False,
              note="Vision Transformer backbone (mmpretrain-compatible, featmap output)"))
register(Rule("SimCCHead", "SimCCHead", _simcc_head, multi_input=True, repeatable=False, is_head=True,
              note="SimCC head: two 1D coordinate classifications per keypoint"))
