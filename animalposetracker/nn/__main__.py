#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: __main__.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Command-line dry-run entry point for the model-building layer.

    Purpose: validate a model yaml without loading torch or allocating device memory
    (checks the from-graph, channel inference, save list and head placement) and print
    a per-layer table.

    Usage:
        # Validate one model
        python -m animalposetracker.nn cfg/models/animalrtpose.yaml -s m

        # Validate every built-in model across all scales
        python -m animalposetracker.nn --all

        # Also instantiate and run a forward pass (requires torch)
        python -m animalposetracker.nn cfg/models/animalrtpose.yaml -s m --build

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

import argparse
import sys
from pathlib import Path

from .spec import describe, parse

SCALES = ("n", "s", "m", "l", "x")


def _check_one(path: Path, scale, build: bool) -> bool:
    """Validate a single yaml; returns True on success."""
    try:
        spec = parse(path, scale=scale)
    except Exception as e:
        print(f"[FAIL] {path.name} scale={scale}  {type(e).__name__}: {e}")
        return False

    print(describe(spec))
    if build:
        try:
            import torch

            from .builder import build_model

            model = build_model(spec)
            params = sum(p.numel() for p in model.parameters())
            print(f"Instantiation OK: {params / 1e6:.2f} M parameters")
            with torch.no_grad():
                out = model(torch.zeros(1, spec.input_ch, 256, 256))
            print(f"Forward OK: output {getattr(out, 'shape', type(out))}")
            print(f"Strides per scale: {model.stride.tolist()}")
        except ImportError:
            print("[SKIP] torch not installed, skipping instantiation and forward")
        except Exception as e:
            print(f"[FAIL] instantiation/forward failed: {type(e).__name__}: {e}")
            return False
    print()
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Model yaml structure check and dry-run")
    p.add_argument("yaml", nargs="?", help="Path to the model yaml")
    p.add_argument("-s", "--scale", default=None, help="Model scale n/s/m/l/x; defaults to the first entry")
    p.add_argument("--all", action="store_true", help="Validate every built-in model across all scales")
    p.add_argument("--build", action="store_true", help="Also instantiate and run a forward pass (requires torch)")
    args = p.parse_args(argv)

    from . import MODEL_YAML_PATHS

    if args.all:
        ok = True
        for name, path in MODEL_YAML_PATHS.items():
            if not Path(path).exists():
                print(f"[SKIP] {name}: file not found {path}")
                continue
            for sc in SCALES:
                ok &= _check_one(Path(path), sc, args.build)
        return 0 if ok else 1

    if not args.yaml:
        p.print_help()
        return 2
    return 0 if _check_one(Path(args.yaml), args.scale, args.build) else 1


if __name__ == "__main__":
    sys.exit(main())
