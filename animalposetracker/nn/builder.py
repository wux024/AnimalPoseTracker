#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: builder.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Instantiate a ModelSpec into a real nn.Module.

    Why a custom container instead of nn.Sequential:
        The from column forms a DAG (multi-input Concat, multi-scale Pose head), which a
        plain nn.Sequential cannot express. GraphSequential keeps intermediate outputs by
        layer index and pulls the ones each layer needs, matching the training-side
        _forward_once logic.

    Attributes attached to every layer (same names as the training side):
        i     layer index
        f     raw from value (-1 means straight-through from the previous layer; a list means multi-input)
        np    parameter count of the layer
        type  layer type name
    External code that relies on these attributes for pruning, fusing or printing the
    structure can be reused as is.

Notes:
    - Requires torch.

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from . import head as _head_mod
from . import modules as _modules
from . import transformer as _transformer_mod
from .registry import get_rule
from .spec import ModelSpec, parse

__all__ = ["build_model", "GraphSequential"]


def _lookup_class(cls_name: str):
    """Resolve a class name from a build rule.

    Two kinds of names are accepted:

    - a dotted torch path, e.g. ``nn.Upsample`` or ``nn.AdaptiveAvgPool2d``. These are
      built-ins that belong to torch, so the yaml keeps the upstream spelling and nothing
      is wrapped locally. The leading ``nn.`` is resolved against ``torch.nn``.
    - a plain name, looked up in modules / transformer / head.

    Raises:
        AttributeError: the name matches neither kind, listing where it was looked for.
    """
    if "." in cls_name:
        parts = cls_name.split(".")
        obj = nn
        if parts[0] != "nn":
            raise AttributeError(
                f"Dotted module name '{cls_name}' must start with 'nn.', e.g. 'nn.Upsample'."
            )
        for part in parts[1:]:
            obj = getattr(obj, part, None)
            if obj is None:
                raise AttributeError(f"torch.nn has no attribute '{part}' (from name '{cls_name}')")
        return obj

    for mod in (_modules, _transformer_mod, _head_mod):
        if hasattr(mod, cls_name):
            return getattr(mod, cls_name)
    raise AttributeError(
        f"Module '{cls_name}' is not implemented in nn.modules / nn.transformer / nn.head, "
        f"and it is not a dotted torch name such as 'nn.Upsample'. "
        f"When adding a module, update both registry.py and this lookup."
    )


class GraphSequential(nn.Module):
    """Container that executes the layer list following the from relationships in the yaml.

    Args:
        model (nn.Sequential): The layer list
        save (list[int]): Layer indices whose outputs must be kept
    """

    def __init__(self, model: nn.Sequential, save):
        super().__init__()
        self.model = model
        self.save = save

    def forward(self, x):
        y = []  # outputs per layer (None for layers not kept)
        for m in self.model:
            if m.f != -1:  # not straight-through, gather inputs by from
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def forward_collect(self, x):
        """Return (final output, {layer index: output}) for visualisation or debugging."""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x, {m.i: y[m.i] for m in self.model if y[m.i] is not None}


def _instantiate(cls, args):
    """Instantiate a module from a rule's resolved arguments.

    Two forms are accepted:

    - a positional list, e.g. ``[c1, c2, k, s]`` -- the default, because a yaml layer is
      written as a plain list and maps onto positional arguments most directly;
    - a single dict, meaning "build it purely from keyword arguments". That form exists for
      modules whose signatures are long and sparse (VisionTransformer has a dozen
      parameters, SimCCHead six), where positional passing would be unreadable and fragile.

    The two forms cannot be confused: a single dict argument is not a valid positional
    argument for any module in this project.
    """
    if len(args) == 1 and isinstance(args[0], dict):
        return cls(**args[0])
    return cls(*args)


def build_model(
    src: Union[str, Path, Dict[str, Any], ModelSpec],
    scale: Optional[str] = None,
    ch: int = 3,
    stride_check: bool = True,
) -> GraphSequential:
    """Build a model from a yaml.

    Args:
        src: yaml path / parsed dict / parsed ModelSpec
        scale: Model scale; defaults to the first entry of scales
        ch: Number of input channels
        stride_check: Run one dummy forward pass to derive per-scale strides and call
            bias_init (matches the training side); set False for a faster build

    Returns:
        GraphSequential whose last layer is the detection / pose head

    Raises:
        ValueError / KeyError: Invalid structure or module name (raised by spec.parse)
    """
    spec = src if isinstance(src, ModelSpec) else parse(src, scale=scale, ch=ch)

    layers = []
    for spec_layer in spec.layers:
        rule = get_rule(spec_layer.module)
        cls = _lookup_class(rule.cls)
        try:
            module = (
                nn.Sequential(*(_instantiate(cls, spec_layer.args) for _ in range(spec_layer.n)))
                if spec_layer.n > 1
                else _instantiate(cls, spec_layer.args)
            )
        except TypeError as e:
            raise TypeError(
                f"Layer {spec_layer.i} ({spec_layer.module}) failed to instantiate: {e}\n"
                f"  yaml from={spec_layer.f_raw} n={spec_layer.repeats_raw}\n"
                f"  resolved args={spec_layer.args} "
                f"(if these differ from what you expect, check the rule for this module in registry.py)"
            ) from e

        module.i, module.f = spec_layer.i, spec_layer.f_raw
        module.type = cls.__name__
        module.np = sum(p.numel() for p in module.parameters())
        layers.append(module)

    model = GraphSequential(nn.Sequential(*layers), save=spec.save)

    # Attach model-level metadata to the model and head for reuse by inference / export
    head = model.model[-1]
    model.nc = spec.nc
    model.kpt_shape = spec.kpt_shape
    model.yaml_spec = spec  # keeps a handle on the source spec
    if hasattr(head, "nc"):
        head.nc = spec.nc
    if hasattr(head, "kpt_shape"):
        head.kpt_shape = spec.kpt_shape
        head.nk = spec.kpt_shape[0] * spec.kpt_shape[1]
    head.legacy = True  # matches the training-side checkpoints (traditional structure)
    head.end2end = False

    if stride_check and hasattr(head, "stride"):
        _compute_strides(model, head, ch)

    return model


def _compute_strides(model: GraphSequential, head: nn.Module, ch: int, size: int = 256) -> None:
    """Derive per-scale strides from one dummy forward pass and run bias_init.

    Mirrors the training-side BaseModel initialization; strides are written back to
    head.stride and model.stride.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, ch, size, size)
        try:
            feats = model.forward_collect(dummy)[1]
        except Exception as e:  # a forward failure usually means the structure is wrong
            raise RuntimeError(f"Dummy forward failed, the model structure is probably invalid: {e}") from e

    # The head consumes the last nl entries of the save list; the head itself is normally
    # not in save, so its own output does not need to be kept.
    outs = None
    saved = [feats[i] for i in model.save if isinstance(feats.get(i), torch.Tensor)]
    if not saved:
        raise RuntimeError("Dummy forward produced no multi-scale features; check the save list and from graph")
    head_feats = saved[-head.nl :] if hasattr(head, "nl") else saved
    head.stride = torch.tensor([size / f.shape[-2] for f in head_feats])
    model.stride = head.stride
    if hasattr(head, "anchors"):
        head.anchors = torch.empty(0)  # deferred until the first inference
    if hasattr(head, "bias_init"):
        head.bias_init()
    if was_training:
        model.train()
