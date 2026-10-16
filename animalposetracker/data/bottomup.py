"""Bottom-up samples for the one-stage (RTMO) head.

The image pipeline is the one from :mod:`animalposetracker.data.pose` -- same
augmentation, same letterbox/rect validation resize, same cache. Only the target
encoding differs: instead of the flat YOLO detection targets, every image carries
an mmpose-style bottom-up sample

    {"gt_instance_labels": {"bboxes", "labels", "keypoints",
                            "keypoints_visible", "areas"},
     "metainfo": {"input_size", "image_path"}}

with boxes and keypoints in *pixel* coordinates of the input canvas (the
normalised coordinates every loader in this project emits are multiplied by the
canvas size here). The head's one-stage assignment and the dynamic coordinate
classifier both work in that space, so the conversion belongs at the very end of
the pipeline.

Instance labels keep their class ids; ``single_cls`` still folds them on request,
and the one-stage head trains either way (one class reproduces the reference
class-agnostic setup).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .pose import (
    PoseTextDataset,
    _assemble_loaders,
    _has_validation_split,
    _read_yaml,
)

__all__ = ["BottomupPoseDataset", "bottomup_collate", "build_bottomup_dataloaders"]


class BottomupPoseDataset(PoseTextDataset):
    """A PoseTextDataset that emits one-stage bottom-up samples per image.

    Everything the base class validates still applies (label formats, OKS
    sigmas, flip index, cache); this subclass only re-encodes each sample after
    the shared pipeline has run.
    """

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        image = sample["images"]
        _, height, width = image.shape
        targets = sample["targets"]

        boxes = targets["boxes"].float()  # normalised cxcywh on the input canvas
        keypoints = targets["keypoints"].float()
        canvas = torch.tensor([width, height], dtype=torch.float32)

        centers = boxes[:, :2] * canvas
        sizes = boxes[:, 2:] * canvas
        bboxes = torch.cat((centers - sizes * 0.5, centers + sizes * 0.5), dim=-1)
        points = keypoints[..., :2] * canvas
        if self.keypoint_dimensions == 3:
            visible = keypoints[..., 2].clamp(0.0, 1.0)
        else:
            visible = torch.ones_like(points[..., 0])
        areas = (sizes[:, 0] * sizes[:, 1]).clamp(min=0.0)

        # The mmpose pipeline filters instances without a single visible keypoint
        # (`FilterAnnotations(by_kpt=True, keep_empty=False)`); the box-side
        # filtering already happened in the augmenter's clip/filter pass.
        if visible.numel():
            keep = visible.sum(dim=-1) >= 1
            bboxes, points, visible, areas = (
                bboxes[keep], points[keep], visible[keep], areas[keep]
            )
            targets["classes"] = targets["classes"][keep]
        return {
            "images": image,
            "targets": {
                "gt_instance_labels": {
                    "bboxes": bboxes,
                    "labels": targets["classes"].long(),
                    "keypoints": points,
                    "keypoints_visible": visible,
                    "areas": areas,
                },
                "metainfo": {
                    "input_size": (width, height),
                    "image_path": targets["image_paths"],
                },
            },
        }


def bottomup_collate(samples) -> Dict[str, Any]:
    """Stack images and keep one sample dict per image.

    The one-stage head consumes the list directly (mmpose's ``batch_data_samples``
    convention); there is no flat target tensor to build.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    images = torch.stack([sample["images"] for sample in samples], dim=0)
    return {"images": images, "targets": [sample["targets"] for sample in samples]}


def build_bottomup_dataloaders(
    data_yaml: Union[str, Path],
    image_size: int,
    batch_size: int,
    validation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 0,
    include_validation: bool = True,
    cache: Union[bool, str] = False,
    fraction: float = 1.0,
    augmentation: Optional[Dict[str, Any]] = None,
    epochs: int = 100,
    close_mosaic: int = 10,
    single_cls: bool = False,
    device: str = "auto",
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Training and validation loaders for the one-stage bottom-up head."""
    config_path = Path(data_yaml).expanduser().resolve()
    train_dataset = BottomupPoseDataset(
        config_path,
        "train",
        image_size,
        cache=cache,
        fraction=fraction,
        seed=seed,
        augmentation=augmentation,
        epochs=epochs,
        close_mosaic=close_mosaic,
        single_cls=single_cls,
        batch_size=batch_size,
    )
    validation_dataset = (
        BottomupPoseDataset(
            config_path,
            "val",
            image_size,
            cache=cache,
            single_cls=single_cls,
            rect=True,
            batch_size=int(validation_batch_size or batch_size),
            stride=32,
        ) if _has_validation_split(_read_yaml(config_path)) else None
    ) if include_validation else None
    return _assemble_loaders(
        train_dataset,
        validation_dataset,
        bottomup_collate,
        batch_size=batch_size,
        validation_batch_size=validation_batch_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
