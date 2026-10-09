#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: spec.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Parse a model yaml into a structured spec (ModelSpec) and infer shapes.

    Design point: reading the yaml and inferring channels / spatial ratios is kept completely
    separate from instantiating nn.Modules.
      - this module does pure arithmetic and graph bookkeeping and does not import torch, so it
        runs in any environment
      - consequently structural errors (bad from index, channel mismatch) are caught before any
        model is built, without touching device memory
      - it also enables a CLI dry-run: inspect the structure without loading torch

    The inference rules mirror the training-side parse_model exactly, so the same yaml yields
    the same structure on both sides.

Notes:
    - pyyaml is imported lazily inside _load_yaml, so passing a dict requires no dependencies.

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .registry import BuildContext, get_rule, make_divisible

__all__ = ["LayerSpec", "ModelSpec", "parse", "describe"]


@dataclass
class LayerSpec:
    """Full spec of one layer.

    Args:
        i: Layer index (0-based)
        f: Input sources, resolved to absolute layer indices
        n: Outer repeat count
        module: Module name as written in the yaml
        args: Resolved arguments, ready to pass to the nn.Module
        c1: Input channels (a list for multi-input layers)
        c2: Output channels
        spatial: Output/input size ratio
        repeats_raw: Raw repeats column from the yaml (kept for reference)
        f_raw: Raw from column from the yaml
    """

    i: int
    f: List[int]
    n: int
    module: str
    args: List[Any]
    c1: Union[int, List[int]]
    c2: int
    spatial: float
    repeats_raw: int = 1
    f_raw: Any = -1

    @property
    def is_multi_input(self) -> bool:
        return isinstance(self.c1, list)

    def __str__(self) -> str:
        return (
            f"[{self.i:>3}] from={str(self.f_raw):<16} n={self.repeats_raw:<3} "
            f"{self.module:<18} c1={str(self.c1):<22} -> c2={self.c2:<5} x{self.spatial}"
        )


@dataclass
class ModelSpec:
    """Full spec of a model.

    Args:
        layers: Per-layer specs
        save: Layer indices whose outputs must be kept (multi-scale head inputs)
        ctx: Build context
        path: Source yaml path
        input_ch: Number of input channels
    """

    layers: List[LayerSpec] = field(default_factory=list)
    save: List[int] = field(default_factory=list)
    ctx: Optional[BuildContext] = None
    path: Optional[Path] = None
    input_ch: int = 3

    @property
    def nc(self) -> int:
        return self.ctx.nc

    @property
    def kpt_shape(self):
        return tuple(self.ctx.kpt_shape)

    @property
    def scale(self) -> Optional[str]:
        return self.ctx.scale

    def head_layer(self) -> Optional[LayerSpec]:
        """Return the last layer spec, which is normally the pose head."""
        return self.layers[-1] if self.layers else None


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a yaml. pyyaml is imported here, so the dict-input path needs no pyyaml."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if "backbone" not in d or "head" not in d:
        raise ValueError(f"Model yaml is missing the backbone/head sections: {path}")
    return d


def _build_namespace(d: Dict[str, Any], ctx: BuildContext) -> Dict[str, Any]:
    """Build the namespace used to resolve bare identifiers.

    Mirrors ultralytics: a bare identifier in the yaml args (such as ``[nc, kpt_shape]``) is
    first looked up in the local scope of the parse function, then falls back to literal
    evaluation. This namespace reproduces that behavior.
    """
    ns = dict(d)
    ns.update(
        {
            "nc": ctx.nc,
            "kpt_shape": ctx.kpt_shape,
            "num_keypoints": ctx.kpt_shape[0],
            "depth": ctx.depth,
            "width": ctx.width,
            "max_channels": ctx.max_channels,
            "scale": ctx.scale,
        }
    )
    return ns


def _resolve_arg(a: Any, ns: Dict[str, Any]) -> Any:
    """Resolve one argument: bare identifiers from the namespace, everything else literal-evaluated."""
    if not isinstance(a, str):
        return a
    if a in ns:
        return ns[a]
    try:
        return ast.literal_eval(a)
    except (ValueError, SyntaxError):
        return a  # keep as is and let instantiation report it rather than swallowing it here


def _resolve_f(f: Any, i: int, n_ch: int) -> List[int]:
    """Resolve relative layer indices into absolute ones.

    ultralytics uses ``x % i`` for this. Since the channel table has exactly i entries when the
    loop reaches layer i, the two are equivalent, but ``x % i`` divides by zero at layer 0.
    Using the channel-table length for negative indexing gives the same behavior and covers layer 0.
    """
    raw = [f] if isinstance(f, int) else list(f)
    out = []
    for x in raw:
        if not isinstance(x, int):
            raise TypeError(f"Layer {i}: from must be an int or list[int], got {type(x).__name__}")
        abs_i = n_ch + x if x < 0 else x
        if not 0 <= abs_i < max(n_ch, 1):
            raise ValueError(f"Layer {i}: from={x} is out of range (channel table length {n_ch})")
        out.append(abs_i)
    return out


def parse(
    src: Union[str, Path, Dict[str, Any]],
    scale: Optional[str] = None,
    ch: int = 3,
) -> ModelSpec:
    """Parse a model yaml (or an already-loaded dict) into a ModelSpec.

    Args:
        src: yaml path, or an already-loaded dict
        scale: Model scale such as 'n'/'s'/'m'/'l'/'x'; None takes the first scales entry
        ch: Number of input channels

    Returns:
        ModelSpec

    Raises:
        ValueError: Invalid structure (bad from index, unknown scale, missing backbone/head, ...)
        KeyError: Unknown module name
    """
    if isinstance(src, dict):
        d, path = src, None
    else:
        path = Path(src)
        d = _load_yaml(path)

    scales = d.get("scales")
    ctx = BuildContext(
        nc=int(d.get("nc", 80)),
        kpt_shape=d.get("kpt_shape", (17, 3)),
        scale=scale,
        scales=scales,
        depth=d.get("depth_multiple", 1.0),
        width=d.get("width_multiple", 1.0),
        top_level=d,
    )
    if scales and scale is not None and scale not in scales:
        raise ValueError(f"scale '{scale}' is not in the yaml scales, available: {sorted(scales)}")

    ns = _build_namespace(d, ctx)

    spec = ModelSpec(ctx=ctx, path=path, input_ch=ch)
    ch_list = [ch]  # same as ultralytics: ch[0] is the input, output channels are appended per layer
    blocks = d["backbone"] + d["head"]

    for i, item in enumerate(blocks):
        if len(item) != 4:
            raise ValueError(f"Layer {i} must be a [from, repeats, module, args] tuple, got {item}")
        f_raw, n_raw, m_name, args_raw = item
        rule = get_rule(m_name)

        # 1) Resolve bare identifiers / literals in the arguments
        args = [_resolve_arg(a, ns) for a in list(args_raw or [])]

        # 2) Resolve from and fetch input channels.
        #    -1 at layer 0 refers to the input image, not to layer 0 itself, so the
        #    forward-reference check does not apply there.
        f = _resolve_f(f_raw, i, len(ch_list))
        if i == 0 and f != [0]:
            raise ValueError(f"Layer 0 can only receive the input image (from=-1), got from={f_raw}")
        for x in f:
            if i > 0 and x >= i:
                raise ValueError(f"Layer {i}: from points at a future layer {x} (forward references are invalid)")
        if rule.multi_input:
            c1 = [ch_list[x] for x in f]
        else:
            if len(f) != 1:
                raise ValueError(f"Module {m_name} only accepts a single input, but from={f_raw}")
            c1 = ch_list[f[0]]

        # 3) Depth scaling (only when n > 1 and the module is repeatable)
        n = max(round(n_raw * ctx.depth), 1) if (n_raw > 1 and rule.repeatable) else n_raw

        # 4) The rule translates the arguments and reports output channels and spatial ratio
        new_args, new_n, c2, spatial = rule.fn(args, n, c1, ctx)

        spec.layers.append(
            LayerSpec(
                i=i,
                f=f,
                n=new_n,
                module=m_name,
                args=new_args,
                c1=c1,
                c2=c2,
                spatial=spatial,
                repeats_raw=n_raw,
                f_raw=f_raw,
            )
        )

        # 5) Record layers that are explicitly referenced by later layers (-1 means straight-through
        #    and is excluded; multi-scale heads need this list).
        #    Equivalent to ultralytics' `x % i`: the channel table has exactly i entries at layer i,
        #    so negative values map to n_ch + x (= x % i) and positive values map to themselves.
        raw_f = [f_raw] if isinstance(f_raw, int) else list(f_raw)
        spec.save.extend(
            len(ch_list) + x if x < 0 else x for x in raw_f if isinstance(x, int) and x != -1
        )
        if i == 0:
            ch_list = []  # layer 0 replaces the input placeholder
        ch_list.append(c2)

    spec.save = sorted(set(x for x in spec.save if x >= 0))

    # 6) A head layer must be last -- the head's "output channels" is only a placeholder and no
    #    later layer should read it
    heads = [l.i for l in spec.layers if get_rule(l.module).is_head]
    if heads and heads[-1] != len(spec.layers) - 1:
        raise ValueError(f"Head layer(s) {heads} must be the last layer; no layer may follow")
    return spec


def describe(spec: ModelSpec, verbose: bool = True) -> str:
    """Render a human-readable structure table, used for dry-run validation."""
    lines = []
    if verbose:
        lines.append(
            f"model: {spec.path.name if spec.path else '<dict>'}  scale={spec.scale}  "
            f"depth={spec.ctx.depth}  width={spec.ctx.width}  max_ch={spec.ctx.max_channels}"
        )
        lines.append(f"nc={spec.nc}  kpt_shape={spec.kpt_shape}  layers={len(spec.layers)}")
        lines.append("-" * 110)
    for layer in spec.layers:
        lines.append(str(layer))
    if verbose:
        lines.append("-" * 110)
        lines.append(f"save (multi-scale head inputs): {spec.save}")
        total = sum(l.n for l in spec.layers)
        lines.append(f"Total layer instances (repeats expanded): {total}")
    return "\n".join(lines)
