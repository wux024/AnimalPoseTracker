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
from typing import Any, Callable, Dict, Optional, Sequence

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
    """Standard / depthwise convolution: channels scaled by width, rounded up to a multiple of 8.

    yaml form ``[c2, k, s, p, ...]``: the stride is the third entry, so the spatial
    ratio comes from ``args[2]``. A missing stride means 1 (kernel-only rows such as
    ``[256, 3]`` keep the resolution).
    """
    c2 = make_divisible(min(args[0], ctx.max_channels) * ctx.width, 8)
    stride = args[2] if len(args) > 2 else 1
    return [ch_in, c2, *args[1:]], n, c2, float(stride)


def _conv_transpose(args, n, ch_in, ctx):
    """Transposed convolution, same argument layout as Conv.

    A stride above 1 upscales here (the opposite of a regular convolution), so the
    spatial ratio is ``1 / stride``.
    """
    c2 = make_divisible(min(args[0], ctx.max_channels) * ctx.width, 8)
    stride = args[2] if len(args) > 2 else 1
    spatial = 1.0 / stride if stride else 1.0
    return [ch_in, c2, *args[1:]], n, c2, spatial


def _sppf(args, n, ch_in, ctx):
    """SPPF: channels scale like Conv, the spatial size never changes.

    yaml form ``[c2, k, n, shortcut]`` (yolo26) or ``[c2]`` (legacy rows). SPPF must
    not go through the Conv rule: its second argument is the pooling kernel, not a
    stride.
    """
    c2 = make_divisible(min(args[0], ctx.max_channels) * ctx.width, 8)
    return [ch_in, c2, *args[1:]], n, c2, 1.0


def _stem(args, n, ch_in, ctx):
    """STEM: channels scaled by width directly (no multiple-of-8 alignment), always downsamples by 2."""
    c2 = int(args[0] * ctx.width)
    return [ch_in, c2], n, c2, 2.0


def _cspnext_block(args, n, ch_in, ctx):
    """CSPNeXtBlock: repeats move into the arguments as the internal bottleneck count, outer repeats set to 1."""
    c2 = int(args[0] * ctx.width)
    return [ch_in, c2, n, *args[1:]], 1, c2, 1.0


def _c3k2(args, n, ch_in, ctx):
    """C3k2 (YOLO26 backbone block): the repeats column moves into the arguments.

    yaml form:
        [c2, c3k, e, attn, g, shortcut]
    Every entry after c2 is optional and positional, matching
    ``C3k2(c1, c2, n, c3k=False, e=0.5, attn=False, g=1, shortcut=True)``.

    Two upstream conventions are kept:
      - c2 is width-scaled with the same rounding as Conv (multiple of 8);
      - scales m / l / x run C3k inner blocks (the ``c3k`` slot is forced True),
        which is why the yolo26 yaml rows are written for the n / s sizes.
    """
    c2 = make_divisible(min(args[0], ctx.max_channels) * ctx.width, 8)
    new_args = [ch_in, c2, *args[1:]]
    new_args.insert(2, n)
    if ctx.scale in ("m", "l", "x"):
        new_args[3:4] = [True]
    return new_args, 1, c2, 1.0


def _c2psa(args, n, ch_in, ctx):
    """C2PSA (YOLO26 attention block): like C3k2 the repeats move into the arguments.

    yaml form:
        [c2, e]
    matching ``C2PSA(c1, c2, n, e=0.5)``.
    """
    c2 = make_divisible(min(args[0], ctx.max_channels) * ctx.width, 8)
    return [ch_in, c2, n, *args[1:]], 1, c2, 1.0


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
    An optional 8th entry is a VisionTransformer keyword-argument mapping, e.g.
    {drop_path_rate: 0.1}.

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
    vit_options = args[7] if len(args) > 7 else None
    if vit_options is not None and not isinstance(vit_options, dict):
        raise TypeError(f"ViT's optional 8th argument must be a mapping, got {type(vit_options).__name__}")

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
    if vit_options:
        allowed_options = {
            "drop_rate",
            "drop_path_rate",
            "qkv_bias",
            "final_norm",
            "out_type",
            "with_cls_token",
        }
        unknown = sorted(set(vit_options) - allowed_options)
        if unknown:
            raise ValueError(f"Unsupported ViT options: {unknown}; allowed: {sorted(allowed_options)}")
        kwargs.update(vit_options)

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


def _hybrid_encoder(args, n, ch_in, ctx):
    """RTMO hybrid-encoder neck: backbone levels in, selected levels out.

    yaml form:
        [hidden_dim, use_encoder_idx, num_encoder_layers, output_indices]

    ``in_channels`` is taken from the incoming feature maps, so the yaml does not
    repeat backbone channel counts. The transformer config and the 1x1 output
    projector are derived from ``hidden_dim``; the CSPRepLayer width / depth
    follow the model scale (ctx.width / ctx.depth), the same way the backbone is
    scaled. The projector travels as a plain dict because this function also runs
    inside spec.parse, which must stay torch-free -- building the module happens
    at instantiation time.

    On c2: the neck emits one feature map per ``output_indices`` entry, all at
    ``hidden_dim`` channels. The channel table holds a single number, so
    hidden_dim is reported (equivalent to ultralytics not rewriting c2; the head
    that follows re-derives what it needs from it).
    """
    hidden_dim = int(args[0])
    use_encoder_idx = list(args[1]) if len(args) > 1 and args[1] else [2]
    num_encoder_layers = int(args[2]) if len(args) > 2 else 1
    output_indices = list(args[3]) if len(args) > 3 and args[3] is not None else None

    in_channels = list(ch_in) if isinstance(ch_in, (list, tuple)) else [int(ch_in)]
    n_out = len(output_indices) if output_indices is not None else len(in_channels)

    kwargs = dict(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        use_encoder_idx=use_encoder_idx,
        num_encoder_layers=num_encoder_layers,
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=hidden_dim, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=hidden_dim, feedforward_channels=hidden_dim * 4, ffn_drop=0.0),
        ),
        output_indices=output_indices,
        projector=dict(in_channels=[hidden_dim] * n_out, out_channels=hidden_dim, kernel_size=1),
        widen_factor=ctx.width,
        deepen_factor=ctx.depth,
    )
    return [kwargs], n, hidden_dim, 1.0


def _rtmo_head(args, n, ch_in, ctx):
    """RTMO one-stage (bottom-up) pose head with the dynamic coordinate classifier.

    yaml form:
        [num_keypoints, featmap_strides, head_module_cfg, dcc_cfg]

    ``head_module_cfg`` writes the widths directly (the effective numbers): the
    head module splits every incoming feature map in half, so ``in_channels`` is
    derived here and must not be repeated, and ``widen_factor`` has no meaning in
    this layout. ``num_classes`` follows the yaml's top-level ``nc`` (the trainer
    sets it from the dataset), and the class count reaches both the head module's
    classification branch and the head's loss. ``dcc_cfg`` configures the dynamic
    coordinate classifier; its input width follows the pose feature
    (``pose_vec_channels``) when one is requested. Everything the head needs at
    runtime -- prior generator, SimOTA assigner and the six loss objects --
    belongs to the training side and is injected there (see training/losses.py
    RTMOLoss), the same split the other two heads use for their loss-side pieces.

    The head consumes the whole multi-scale output of the preceding layer (the
    neck) as a single item, so ``multi_input`` is False.

    On c2: the head emits decoded predictions, not a feature map; the incoming
    channel count is only echoed to keep the channel table well-formed. The head
    must be the last layer, so nothing reads it.
    """
    if len(args) < 4:
        raise ValueError(
            f"RTMOHead needs [num_keypoints, featmap_strides, head_module_cfg, dcc_cfg], got {args}"
        )
    num_keypoints = int(args[0])
    featmap_strides = tuple(int(stride) for stride in args[1])
    head_module_cfg = dict(args[2] or {})
    dcc_cfg = dict(args[3] or {})

    if isinstance(ch_in, (list, tuple)) and len(set(ch_in)) != 1:
        raise ValueError(
            f"The RTMO head splits every incoming feature map in half, so all of them must "
            f"share one channel count; got {ch_in}"
        )
    channels = ch_in[-1] if isinstance(ch_in, (list, tuple)) else int(ch_in)
    if channels % 2:
        raise ValueError(f"The RTMO head needs an even number of input channels, got {channels}")
    if "in_channels" in head_module_cfg:
        raise ValueError(
            "RTMOHead derives head_module_cfg['in_channels'] from the neck; remove the explicit value"
        )
    if float(head_module_cfg.get("widen_factor", 1.0)) != 1.0:
        raise ValueError(
            "RTMOHead widths are written as effective numbers; "
            "drop widen_factor and write the final channel counts"
        )
    if "num_classes" in head_module_cfg:
        raise ValueError(
            "RTMOHead derives head_module_cfg['num_classes'] from the yaml's nc; "
            "remove the explicit value"
        )

    half = channels // 2
    head_module_cfg["in_channels"] = half
    head_module_cfg["num_classes"] = int(ctx.nc)
    pose_vec_channels = int(head_module_cfg.get("pose_vec_channels", -1))
    dcc_cfg.setdefault("in_channels", pose_vec_channels if pose_vec_channels > 0 else half)

    kwargs = dict(
        num_keypoints=num_keypoints,
        head_module_cfg=head_module_cfg,
        featmap_strides=featmap_strides,
        num_classes=int(ctx.nc),
        dcc_cfg=dcc_cfg,
    )
    return [kwargs], n, channels, 1.0


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
register(Rule("ConvTranspose", "ConvTranspose", _conv_transpose, note="Transposed convolution (stride > 1 upscales)"))
register(Rule("Concat", "Concat", _concat, multi_input=True, note="Multi-input concat, channels summed"))
register(Rule("SPPF", "SPPF", _sppf, note="Spatial pyramid pooling (kernel args are not strides)"))
register(Rule("nn.Upsample", "nn.Upsample", _upsample, repeatable=False,
              note="Native torch upsampling, built directly from torch.nn"))
register(Rule("STEM", "STEM", _stem, note="Custom: Spatial-Channel Excitation"))
register(Rule("CSPNeXtBottleneck", "CSPNeXtBottleneck", _passthrough, note="Custom: CSPNeXt bottleneck"))
register(Rule("CSPNeXtBlock", "CSPNeXtBlock", _cspnext_block, note="Custom: repeats move into args"))
register(Rule("YOLOPoseHead", "YOLOPoseHead", _pose_head, multi_input=True, repeatable=False, is_head=True,
              note="Anchor-based pose head (boxes + classes + keypoints)"))
register(Rule("ViT", "VisionTransformer", _vit, repeatable=False,
              note="Vision Transformer backbone (mmpretrain-compatible, featmap output)"))
register(Rule("SimCCHead", "SimCCHead", _simcc_head, multi_input=True, repeatable=False, is_head=True,
              note="SimCC head: two 1D coordinate classifications per keypoint"))
register(Rule("C3k2", "C3k2", _c3k2, note="YOLO26 block: repeats move into args, m/l/x force c3k inner blocks"))
register(Rule("C2PSA", "C2PSA", _c2psa, note="YOLO26 attention block: repeats move into args"))
register(Rule("HybridEncoder", "HybridEncoder", _hybrid_encoder, multi_input=True, repeatable=False,
              note="RTMO neck: transformer encoder + FPN/PAN; multi-level in, selected levels out"))
register(Rule("RTMOHead", "RTMOHead", _rtmo_head, repeatable=False, is_head=True,
              note="RTMO one-stage pose head (bottom-up, dynamic coordinate classifier)"))
