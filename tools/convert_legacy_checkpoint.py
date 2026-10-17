#!/usr/bin/env python
"""One-time converter: legacy object-style checkpoints -> the project's plain dict format.

The training code stores plain ``{"model_state_dict": tensors}`` payloads, so
``weights_only=True`` reads them without any help. Some historical files (and
third-party exports) instead carry serialized model *objects* whose class paths
live in another package. This tool restores those objects against this
project's own classes, extracts the tensors, and writes the plain format.

Usage:
    python tools/convert_legacy_checkpoint.py path/to/legacy.pt [output.pt]

The default output is ``<name>-weights.pt`` next to the source file.
"""

import argparse
import importlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from animalposetracker.training.checkpoint import _numpy_checkpoint_globals  # noqa: E402


def _aliases():
    """Class-path aliases found in historical checkpoints.

    Maps the name recorded inside the file to the equivalent class of this
    project, so the safe loader can rebuild the serialized objects.

    Two families are covered: ultralytics pose checkpoints (the ones the project
    trains against) and ultralytics detection checkpoints (the YOLO26 line, whose
    backbone this project reuses -- converting one gives the AnimalRTMO backbone a
    COCO-trained initialization, the same recipe mmpose follows with YOLOX). A
    detection checkpoint's ``Detect`` head has no counterpart here, so it is
    rebuilt against a bare module: its tensors convert but nothing ever loads them.
    """
    import torch.nn as nn

    from animalposetracker.nn.attention import Attention, ChannelAttention
    from animalposetracker.nn.block import RealNVP
    from animalposetracker.nn.builder import GraphSequential
    from animalposetracker.nn.head import YOLOPoseHead
    from animalposetracker.nn.block import (
        Bottleneck,
        C2PSA,
        C3k,
        C3k2,
        CSPNeXtBlock,
        CSPNeXtBottleneck,
        Concat,
        DFL,
        PSABlock,
        SPPF,
        STEM,
    )
    from animalposetracker.nn.conv import Conv, DWConv

    table = {
        "ultralytics.nn.tasks.PoseModel": GraphSequential,
        "ultralytics.nn.modules.block.STEM": STEM,
        "ultralytics.nn.modules.conv.Conv": Conv,
        "ultralytics.nn.modules.block.CSPNeXtBlock": CSPNeXtBlock,
        "ultralytics.nn.modules.block.CSPNeXtBottleneck": CSPNeXtBottleneck,
        "ultralytics.nn.modules.conv.DWConv": DWConv,
        "ultralytics.nn.modules.conv.ChannelAttention": ChannelAttention,
        "ultralytics.nn.modules.block.SPPF": SPPF,
        "ultralytics.nn.modules.conv.Concat": Concat,
        "ultralytics.nn.modules.head.Pose": YOLOPoseHead,
        "ultralytics.nn.modules.block.DFL": DFL,
        # The YOLO26 pose family shares the backbone with the detection models, so its
        # checkpoints initialize the one-stage backbone the same way; only the head
        # class name is new (Pose26 / the RLE flow).
        "ultralytics.nn.modules.head.Pose26": YOLOPoseHead,
        "ultralytics.nn.modules.head.RealNVP": RealNVP,
        # Detection family (YOLO26): same blocks plus the Detect head.
        "ultralytics.nn.tasks.DetectionModel": GraphSequential,
        "ultralytics.nn.modules.block.C3k2": C3k2,
        "ultralytics.nn.modules.block.C3k": C3k,
        "ultralytics.nn.modules.block.C2PSA": C2PSA,
        "ultralytics.nn.modules.block.PSABlock": PSABlock,
        "ultralytics.nn.modules.block.Attention": Attention,
        "ultralytics.nn.modules.block.Bottleneck": Bottleneck,
        "ultralytics.nn.modules.head.Detect": nn.Module,
        # Classification family (ImageNet): its graph and head classes.
        "ultralytics.nn.tasks.ClassificationModel": GraphSequential,
        "ultralytics.nn.modules.head.Classify": nn.Module,
    }
    aliases = []
    for qualified_name, local_class in table.items():
        module_name, _, class_name = qualified_name.rpartition(".")
        aliases.append(type(class_name, (local_class,), {"__module__": module_name}))
    try:
        from torchvision import transforms as _torchvision_transforms
        from torchvision.transforms import functional as _torchvision_functional
    except (ImportError, RuntimeError):  # torchvision is optional; only checkpoints carry it
        _torchvision_transforms = None
        _torchvision_functional = None
    if _torchvision_transforms is not None:
        # Classification checkpoints serialize their preprocessing pipeline. The classes
        # are installed here, so the real ones are allowlisted (an enum such as
        # InterpolationMode cannot be subclassed); they are never executed.
        for name in ("Compose", "Resize", "CenterCrop", "Normalize", "ToTensor"):
            local = getattr(_torchvision_transforms, name, None)
            if local is not None:
                aliases.append(local)
    if _torchvision_functional is not None:
        interpolation = getattr(_torchvision_functional, "InterpolationMode", None)
        if interpolation is not None:
            aliases.append(interpolation)
    return aliases


def _state_from_payload(payload):
    if not isinstance(payload, dict):
        raise SystemExit("File does not contain a state dict or model checkpoint")
    for key in ("model_state_dict", "ema_state_dict", "state_dict", "model", "ema", "weights"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, torch.nn.Module):
            return candidate.state_dict()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="legacy checkpoint to convert")
    parser.add_argument("output", nargs="?", help="destination (default: <name>-weights.pt)")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"File does not exist: {source}")
    destination = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}-weights{source.suffix}")
    )

    allowed = _aliases() + _numpy_checkpoint_globals() + [set]
    seen = set()
    payload = None
    for _ in range(64):
        try:
            with torch.serialization.safe_globals(allowed):
                payload = torch.load(str(source), map_location="cpu", weights_only=True)
            break
        except Exception as exc:
            match = re.search(r"Unsupported global: GLOBAL ([A-Za-z0-9_.]+)", str(exc))
            if match is None:
                raise SystemExit(f"Could not read {source.name}: {exc}") from exc
            qualified = match.group(1)
            if qualified in seen:
                raise SystemExit(f"Could not read {source.name}: repeats {qualified}")
            seen.add(qualified)
            if qualified.startswith("torch.nn."):
                module_name, _, class_name = qualified.rpartition(".")
                module = importlib.import_module(module_name)
                allowed.append(getattr(module, class_name))
                continue
            raise SystemExit(
                f"Could not read {source.name}: serialized class {qualified} is not covered. "
                "If it maps to a project class, add it to _aliases()."
            ) from exc
    else:
        raise SystemExit("Too many serialized classes for conversion")
    assert payload is not None

    state = {name: value for name, value in _state_from_payload(payload).items() if torch.is_tensor(value)}
    if not state:
        raise SystemExit("No tensors found in the checkpoint")

    torch.save({"format_version": 1, "model_state_dict": state}, destination)
    print(f"wrote {destination} ({len(state)} tensors)")

    check = torch.load(str(destination), map_location="cpu", weights_only=True)
    assert len(check["model_state_dict"]) == len(state)
    print("verified: the plain format reloads without aliases")


if __name__ == "__main__":
    main()
