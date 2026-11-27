#!/usr/bin/env python3
"""Export a standard backbone as the embedding ONNX the ReID encoder expects.

``OnnxReIDEncoder`` speaks one narrow contract: input ``(N, 3, H, W)`` in 0..1, output a
single ``(N, D)`` vector. torchvision and torch.hub backbones do not honour it -- they
want ImageNet normalisation, and some return a dict or a 4-D feature map. This tool wraps
one backbone into that contract once, offline. It is not a runtime component.

Each backbone is one entry in ``BACKBONES``: how to load it, how it normalises, and how
its output becomes a vector. There is deliberately no architecture guessing here -- a new
model means a new entry, not new logic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Backbone:
    """How to turn one pretrained model into 'image in, vector out'."""

    loader: Callable[[], nn.Module]
    mean: tuple
    std: tuple
    take: Callable[[object], torch.Tensor]


def _torchvision_headless(builder_name: str, weights_enum: str, member: str):
    """Load a classifier and cut its final layer, keeping the pooled features.

    torchvision is imported here, not at module scope: this tool is run once per model on
    a workstation that already has it, while the package itself never needs it. The
    weights are a two-step lookup because the enum lives inside torchvision.models --
    ``getattr`` does not walk dots.
    """

    def load():
        import torchvision.models as models

        builder = getattr(models, builder_name)
        weights = getattr(getattr(models, weights_enum), member)
        model = builder(weights=weights)
        return nn.Sequential(*list(model.children())[:-1])

    return load


def _dinov2(name):
    def load():
        return torch.hub.load("facebookresearch/dinov2", name)

    return load


BACKBONES = {
    # ViT, self-supervised: the CLS token is the embedding, no pooling needed.
    "dinov2_vits14": Backbone(
        loader=_dinov2("dinov2_vits14"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        take=lambda out: out["x_norm_clstoken"],
    ),
    "dinov2_vitb14": Backbone(
        loader=_dinov2("dinov2_vitb14"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        take=lambda out: out["x_norm_clstoken"],
    ),
    # CNN, supervised: the head-less network already ends in a global average pool.
    "resnet50": Backbone(
        loader=_torchvision_headless("resnet50", "ResNet50_Weights", "IMAGENET1K_V2"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        take=lambda out: out.flatten(1),
    ),
    "convnext_tiny": Backbone(
        loader=_torchvision_headless("convnext_tiny", "ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        take=lambda out: out.flatten(1),
    ),
}


class Wrapped(nn.Module):
    """Normalise inside the graph so the contract's 0..1 input stays the boundary."""

    def __init__(self, spec: Backbone) -> None:
        super().__init__()
        self.backbone = spec.loader().eval()
        self.register_buffer("mean", torch.tensor(spec.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec.std).view(1, 3, 1, 1))
        self.take = spec.take

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalised = (images - self.mean) / self.std
        return self.take(self.backbone(normalised))


def export(name: str, output: Path, imgsz: int) -> None:
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name!r}; choose one of {sorted(BACKBONES)}")
    model = Wrapped(BACKBONES[name]).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        embedding = model(dummy)
    if embedding.dim() != 2:
        raise SystemExit(
            f"{name} produced {tuple(embedding.shape)}; the ReID contract needs (N, D)"
        )
    print(f"{name}: input (1, 3, {imgsz}, {imgsz}) -> embedding {tuple(embedding.shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["embedding"],
        # Batch is dynamic because the tracker encodes every box of a frame in one call;
        # spatial axes stay fixed so the encoder can read its crop size off the graph.
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {output}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--output", required=True, help="Destination .onnx path")
    parser.add_argument("--imgsz", type=int, default=224, help="Square input size")
    args = parser.parse_args(argv)
    export(args.backbone, Path(args.output).expanduser(), int(args.imgsz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
