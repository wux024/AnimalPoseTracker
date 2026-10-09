#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: __init__.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    AnimalPoseTracker model-building layer (yaml driven).

    Usage:
        from animalposetracker.nn import build_model, MODEL_YAML_PATHS

        model = build_model(MODEL_YAML_PATHS["AnimalRTPose"], scale="m")
        out = model(torch.zeros(1, 3, 256, 256))

    Layers:
        registry.py  module name -> build rule (pure python, no torch)
        spec.py      yaml -> ModelSpec + shape inference (pure python; pyyaml only when reading files)
        modules.py   base layers and custom modules (torch)
        head.py      Pose head (torch)
        builder.py   ModelSpec -> nn.Module (torch)

    Scope:
        This layer only defines and instantiates network structures. The training loop,
        data pipeline, losses, evaluation and export live elsewhere. The yaml syntax
        matches the training side so a single yaml can be read by both, keeping the
        resulting structure identical.

Notes:
    - No dependency on ultralytics / mmcv / mmengine.
    - spec / registry do not import torch, so yaml can be validated without torch installed.

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

from animalposetracker.cfg import MODEL_DIR, MODEL_YAML_PATHS

__all__ = [
    "MODEL_YAML_PATHS",
    "MODEL_DIR",
    "parse",
    "describe",
    "build_model",
    "GraphSequential",
    "RULES",
    "rule_names",
]

_LAZY = {
    # public name: (submodule, attribute)
    "parse": ("spec", "parse"),
    "describe": ("spec", "describe"),
    "ModelSpec": ("spec", "ModelSpec"),
    "LayerSpec": ("spec", "LayerSpec"),
    "build_model": ("builder", "build_model"),
    "GraphSequential": ("builder", "GraphSequential"),
    "RULES": ("registry", "RULES"),
}


def rule_names():
    """List all registered module names (torch-free, usable for self-checks)."""
    from .registry import RULES

    return sorted(RULES)


def __getattr__(name):
    """Import on demand so that importing this package does not require torch."""
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(f".{mod_name}", __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
