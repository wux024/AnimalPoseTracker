"""MMPose-compatible top-down data, SimCC loss, and validation adapters.

The adapters in this module plug into the shared AnimalPoseTracker ``Trainer``. They mirror
the AnimalViTPose MMPose recipe without introducing a second training loop or a runtime
dependency on MMPose.
"""

import hashlib
import io
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import truncnorm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .data import PoseTextDataset, _read_yaml
from .metrics import build_coco_ground_truth, evaluate_coco_keypoints, keypoint_error_metrics


class SimCCLabel:
    """Encode keypoints as the 1D Gaussian labels used by MMPose ``SimCCLabel``."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        sigma: float = 6.0,
        split_ratio: float = 2.0,
        normalize: bool = True,
    ) -> None:
        self.input_size = tuple(map(int, input_size))
        self.sigma = float(sigma)
        self.split_ratio = float(split_ratio)
        self.normalize = bool(normalize)
        if len(self.input_size) != 2 or min(self.input_size) < 1:
            raise ValueError("SimCC input_size must be a positive (width, height) pair")
        if self.sigma <= 0 or self.split_ratio <= 0:
            raise ValueError("SimCC sigma and split_ratio must be positive")

    def encode(self, keypoints: np.ndarray, visible: np.ndarray):
        """Return x/y labels and visibility weights for one cropped instance."""
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        weights = np.asarray(visible, dtype=np.float32).reshape(-1).copy()
        if len(weights) != len(keypoints):
            raise ValueError("SimCC keypoint coordinates and visibility have different lengths")

        width, height = self.input_size
        bins_x = int(np.around(width * self.split_ratio))
        bins_y = int(np.around(height * self.split_ratio))
        labels_x = np.zeros((len(keypoints), bins_x), dtype=np.float32)
        labels_y = np.zeros((len(keypoints), bins_y), dtype=np.float32)
        split_points = np.around(keypoints * self.split_ratio).astype(np.int64)
        x_grid = np.arange(bins_x, dtype=np.float32)
        y_grid = np.arange(bins_y, dtype=np.float32)
        radius = self.sigma * 3.0

        for index, (mu_x, mu_y) in enumerate(split_points):
            if weights[index] < 0.5:
                continue
            if mu_x < 0 or mu_y < 0 or mu_x >= bins_x or mu_y >= bins_y:
                weights[index] = 0.0
                continue
            left, top = mu_x - radius, mu_y - radius
            right, bottom = mu_x + radius + 1, mu_y + radius + 1
            if left >= bins_x or top >= bins_y or right < 0 or bottom < 0:
                weights[index] = 0.0
                continue
            labels_x[index] = np.exp(-((x_grid - mu_x) ** 2) / (2.0 * self.sigma**2))
            labels_y[index] = np.exp(-((y_grid - mu_y) ** 2) / (2.0 * self.sigma**2))

        if self.normalize:
            norm = self.sigma * np.sqrt(2.0 * np.pi)
            labels_x /= norm
            labels_y /= norm
        return labels_x, labels_y, weights


class SimCCKLLoss:
    """MMPose ``KLDiscretLoss`` defaults for the AnimalViTPose SimCC head."""

    # MMPose sums the per-instance KL terms and divides by K, not by batch size.
    loss_is_batch_sum = True

    def __init__(self, keypoint_count: int, beta: float = 1.0) -> None:
        self.keypoint_count = int(keypoint_count)
        self.beta = float(beta)
        if self.keypoint_count < 1 or self.beta <= 0:
            raise ValueError("keypoint_count and KL beta must be positive")

    def __call__(self, predictions, targets: Dict[str, torch.Tensor]):
        if not isinstance(predictions, (tuple, list)) or len(predictions) != 2:
            raise TypeError("SimCCHead must return the x and y coordinate logits")
        pred_x, pred_y = predictions
        target_x = targets["simcc_x"]
        target_y = targets["simcc_y"]
        weights = targets["keypoint_weights"].reshape(-1)
        if pred_x.shape != target_x.shape or pred_y.shape != target_y.shape:
            raise ValueError(
                "SimCC prediction/target shapes differ: "
                f"x={tuple(pred_x.shape)}/{tuple(target_x.shape)}, "
                f"y={tuple(pred_y.shape)}/{tuple(target_y.shape)}"
            )
        if pred_x.shape[1] != self.keypoint_count or pred_y.shape[1] != self.keypoint_count:
            raise ValueError("SimCC output keypoint count does not match the dataset")

        def axis_loss(prediction, target):
            logits = prediction.reshape(-1, prediction.shape[-1]) * self.beta
            labels = target.reshape(-1, target.shape[-1])
            log_probability = F.log_softmax(logits, dim=1)
            divergence = F.kl_div(log_probability, labels, reduction="none").mean(dim=1)
            return (divergence * weights).sum() / self.keypoint_count

        loss_x = axis_loss(pred_x, target_x)
        loss_y = axis_loss(pred_y, target_y)
        return {
            "loss": loss_x + loss_y,
            "loss_simcc_x": loss_x,
            "loss_simcc_y": loss_y,
        }


def _rotate_point(point: np.ndarray, angle_radians: float) -> np.ndarray:
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    return np.asarray(
        [point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine],
        dtype=np.float32,
    )


def _third_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = first - second
    return second + np.asarray([-direction[1], direction[0]], dtype=np.float32)


def _topdown_warp_matrix(center, scale, rotation, output_size):
    """Port MMPose's ``get_warp_matrix`` for the default non-UDP affine transform."""
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    output_width, output_height = map(int, output_size)
    angle = np.deg2rad(float(rotation))
    source_direction = _rotate_point(np.asarray([-scale[0] * 0.5, 0.0]), angle)
    target_direction = np.asarray([-output_width * 0.5, 0.0], dtype=np.float32)

    source = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    source[2] = _third_point(source[0], source[1])

    target_center = np.asarray([output_width * 0.5, output_height * 0.5], dtype=np.float32)
    target = np.zeros((3, 2), dtype=np.float32)
    target[0] = target_center
    target[1] = target_center + target_direction
    target[2] = _third_point(target[0], target[1])
    return cv2.getAffineTransform(source, target)


def _fix_aspect_ratio(scale: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    width, height = map(float, scale)
    output_width, output_height = input_size
    aspect_ratio = output_width / output_height
    if width > height * aspect_ratio:
        height = width / aspect_ratio
    else:
        width = height * aspect_ratio
    return np.asarray([width, height], dtype=np.float32)


class TopDownPoseDataset(Dataset):
    """Read one person/animal instance per sample and apply MMPose top-down geometry."""

    PIXEL_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    PIXEL_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(
        self,
        data_yaml: Union[str, Path],
        split: str,
        input_size: Tuple[int, int] = (256, 256),
        sigma: float = 6.0,
        split_ratio: float = 2.0,
        cache: Union[bool, str] = False,
        fraction: float = 1.0,
        seed: int = 0,
        augmentation_config: Optional[Dict[str, Any]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        from .profiles import ANIMALVITPOSE_AUGMENTATION, ANIMALVITPOSE_PREPROCESSING

        self.config_path = Path(data_yaml).expanduser().resolve()
        self.data_config = _read_yaml(self.config_path)
        self.augmentation_config = dict(ANIMALVITPOSE_AUGMENTATION)
        self.augmentation_config.update(augmentation_config or {})
        self.preprocessing_config = dict(ANIMALVITPOSE_PREPROCESSING)
        self.preprocessing_config.update(preprocessing_config or {})
        self.split = str(split)
        self.training = self.split == "train"
        self.input_size = tuple(map(int, input_size))
        if len(self.input_size) != 2 or min(self.input_size) < 1:
            raise ValueError("AnimalViTPose input_size must be a positive (width, height) pair")
        self.codec = SimCCLabel(self.input_size, sigma=sigma, split_ratio=split_ratio)

        # Reuse the shared annotation parser; this wrapper changes only how instances are
        # sampled and geometrically prepared for the SimCC head.
        self.annotations = PoseTextDataset(
            self.config_path,
            split=self.split,
            image_size=max(self.input_size),
            cache=False,
            fraction=fraction if self.training else 1.0,
            seed=seed,
            augmentation=None,
            single_cls=False,
        )
        self.num_keypoints = self.annotations.num_keypoints
        self.keypoint_dimensions = self.annotations.keypoint_dimensions
        self.flip_indices = (
            np.asarray(self.annotations.flip_idx, dtype=np.int64)
            if self.annotations.flip_idx is not None else None
        )
        self.kpt_oks_sigmas = np.asarray(
            self.annotations.kpt_oks_sigmas, dtype=np.float32
        ).reshape(-1)
        if (
            len(self.kpt_oks_sigmas) != self.num_keypoints
            or not np.isfinite(self.kpt_oks_sigmas).all()
            or np.any(self.kpt_oks_sigmas <= 0)
        ):
            raise ValueError(
                "Dataset kpt_oks_sigmas must contain one finite positive value per keypoint"
            )
        if self.flip_indices is not None and sorted(self.flip_indices.tolist()) != list(
            range(self.num_keypoints)
        ):
            raise ValueError("flip_idx must be a permutation of keypoint indices")

        body_names = self.data_config.get("keypoint_names") or self.data_config.get("keypoints")
        if isinstance(body_names, dict):
            body_names = list(body_names.values())
        keypoint_types = self.data_config.get("keypoint_types")
        keypoint_info = self.data_config.get("keypoint_info")
        if isinstance(keypoint_info, dict):
            ordered_info = [keypoint_info[key] for key in sorted(keypoint_info, key=lambda value: int(value))]
            if not body_names:
                body_names = [item.get("name", str(index)) for index, item in enumerate(ordered_info)]
            if keypoint_types is None:
                keypoint_types = [item.get("type", "") for item in ordered_info]
        if not body_names and self.annotations.annotation_path is not None:
            try:
                with self.annotations.annotation_path.open("r", encoding="utf-8-sig") as stream:
                    categories = json.load(stream).get("categories", [])
                body_names = next(
                    (category.get("keypoints") for category in categories if category.get("keypoints")),
                    [],
                )
            except (OSError, ValueError, TypeError):
                body_names = []
        self.keypoint_names = list(body_names or [])
        if isinstance(keypoint_types, dict):
            keypoint_types = [keypoint_types.get(index, keypoint_types.get(str(index), ""))
                              for index in range(self.num_keypoints)]
        if keypoint_types is not None and len(keypoint_types) != self.num_keypoints:
            raise ValueError("keypoint_types must contain one 'upper'/'lower' entry per keypoint")
        inferred_upper = (
            [index for index, kind in enumerate(keypoint_types) if str(kind).casefold() == "upper"]
            if keypoint_types is not None else None
        )
        inferred_lower = (
            [index for index, kind in enumerate(keypoint_types) if str(kind).casefold() == "lower"]
            if keypoint_types is not None else None
        )
        self.upper_body_ids = self._resolve_body_ids(
            self.data_config.get("upper_body_ids", self.data_config.get("upper_body", inferred_upper))
        )
        self.lower_body_ids = self._resolve_body_ids(
            self.data_config.get("lower_body_ids", self.data_config.get("lower_body", inferred_lower))
        )

        self.cache_mode = "ram" if cache is True else str(cache).strip().lower() if cache else "none"
        if self.cache_mode not in {"none", "ram", "disk"}:
            raise ValueError("cache must be false, true, 'ram' or 'disk'")
        self._ram_cache = {}
        self.disk_cache_dir = None
        if self.cache_mode == "disk":
            cache_key = hashlib.sha1(
                f"{self.config_path}|{self.split}|{self.input_size}".encode("utf-8")
            ).hexdigest()[:16]
            self.disk_cache_dir = (
                Path(tempfile.gettempdir()) / "AnimalPoseTracker" / "topdown-cache" / cache_key
            )
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        self._entries = []
        self._image_sizes = {}
        self._image_ids = {}
        next_annotation_id = 1
        for image_index, image_path in enumerate(self.annotations.image_paths):
            image_path = Path(image_path).resolve()
            with Image.open(image_path) as image_header:
                image_width, image_height = image_header.size
            self._image_sizes[str(image_path)] = (image_width, image_height)

            if self.annotations._coco_annotations_by_path is not None:
                image_records = self.annotations._coco_annotations_by_path.get(str(image_path), [])
                for record in image_records:
                    if int(record.get("category_id", -1)) not in self.annotations._coco_category_to_class:
                        continue
                    bbox = np.asarray(record.get("bbox", ()), dtype=np.float32)
                    if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    x1 = float(np.clip(bbox[0], 0, image_width))
                    y1 = float(np.clip(bbox[1], 0, image_height))
                    x2 = float(np.clip(bbox[0] + bbox[2], 0, image_width))
                    y2 = float(np.clip(bbox[1] + bbox[3], 0, image_height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    points, visible = self._coco_keypoints(record)
                    self._entries.append({
                        "image_path": str(image_path),
                        "bbox": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                        "keypoints": points,
                        "visible": visible,
                        "image_id": int(record["image_id"]),
                        "category_id": int(record["category_id"]),
                        "annotation_id": int(record.get("id", next_annotation_id)),
                        "area": float(record.get("area", (x2 - x1) * (y2 - y1))),
                    })
                    next_annotation_id += 1
            else:
                classes, boxes, keypoints = self.annotations._read_targets(
                    image_path, image_width, image_height
                )
                self._image_ids[str(image_path)] = image_index + 1
                for instance_index, (class_id, box, points) in enumerate(zip(classes, boxes, keypoints)):
                    center_x, center_y, box_width, box_height = map(float, box)
                    x1 = (center_x - box_width * 0.5) * image_width
                    y1 = (center_y - box_height * 0.5) * image_height
                    x2 = (center_x + box_width * 0.5) * image_width
                    y2 = (center_y + box_height * 0.5) * image_height
                    visible = points[:, 2] > 0 if points.shape[1] >= 3 else np.ones(
                        self.num_keypoints, dtype=bool
                    )
                    self._entries.append({
                        "image_path": str(image_path),
                        "bbox": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                        "keypoints": points[:, :2] * np.asarray(
                            [image_width, image_height], dtype=np.float32
                        ),
                        "visible": visible.astype(np.float32),
                        "image_id": image_index + 1,
                        "category_id": int(class_id) + 1,
                        "annotation_id": next_annotation_id + instance_index,
                        "area": float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)),
                    })
                next_annotation_id += len(boxes)

        if not self._entries:
            raise ValueError(f"The {self.split} split contains no annotated top-down instances")
        self.annotation_path = self.annotations.annotation_path
        self.class_names = self.annotations.source_class_names
        self.annotation_format = "coco" if self.annotation_path is not None else "yolo"

    def _resolve_body_ids(self, values):
        if values is None:
            return []
        if isinstance(values, (str, int)):
            values = [values]
        name_to_index = {str(name).casefold(): i for i, name in enumerate(self.keypoint_names)}
        result = []
        for value in values:
            if isinstance(value, str) and not value.strip().isdigit():
                if value.casefold() not in name_to_index:
                    raise ValueError(f"Unknown keypoint name in body group: {value}")
                result.append(name_to_index[value.casefold()])
            else:
                result.append(int(value))
        if any(index < 0 or index >= self.num_keypoints for index in result):
            raise ValueError("upper/lower body keypoint index is outside the dataset keypoint range")
        return result

    def _coco_keypoints(self, record):
        values = record.get("keypoints") or []
        points = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        visible = np.zeros(self.num_keypoints, dtype=np.float32)
        if values:
            if len(values) != self.num_keypoints * 3:
                raise ValueError(
                    f"Expected {self.num_keypoints * 3} COCO keypoint values, got {len(values)}"
                )
            points3 = np.asarray(values, dtype=np.float32).reshape(self.num_keypoints, 3)
            if not np.isfinite(points3).all():
                raise ValueError("COCO keypoint annotations must be finite")
            points = points3[:, :2]
            visible = (points3[:, 2] > 0).astype(np.float32)
        return points, visible

    def __len__(self):
        return len(self._entries)

    def set_epoch(self, _epoch: int) -> None:
        """Keep the common trainer's dataset/sampler interface."""

    def _cache_path(self, image_path: str) -> Path:
        source = Path(image_path)
        stat = source.stat()
        key = hashlib.sha1(
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        return self.disk_cache_dir / f"{key}.npy"

    def _read_image(self, image_path: str) -> np.ndarray:
        if self.cache_mode == "ram" and image_path in self._ram_cache:
            return self._ram_cache[image_path].copy()
        cache_path = self._cache_path(image_path) if self.cache_mode == "disk" else None
        if cache_path is not None and cache_path.is_file():
            try:
                image = np.load(cache_path, allow_pickle=False)
                if image.ndim == 3 and image.shape[2] == 3:
                    return image
            except (OSError, ValueError):
                cache_path.unlink(missing_ok=True)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if self.cache_mode == "ram":
            self._ram_cache[image_path] = image.copy()
        elif cache_path is not None:
            temporary = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
            try:
                np.save(temporary, image)
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return image

    def _apply_half_body(self, center, scale, keypoints, visible):
        half_body_probability = float(
            self.augmentation_config.get("half_body_probability", 0.3)
        )
        if (
            (not self.upper_body_ids and not self.lower_body_ids)
            or int(visible.sum()) < 9
            or np.random.rand() >= half_body_probability
        ):
            return center, scale
        upper = [i for i in self.upper_body_ids if visible[i] > 0]
        lower = [i for i in self.lower_body_ids if visible[i] > 0]
        if len(upper) < 2 and len(lower) < 3:
            return center, scale
        if len(lower) < 3:
            selected = upper
        elif len(upper) < 2:
            selected = lower
        else:
            upper_probability = float(
                self.augmentation_config.get("half_body_upper_probability", 0.7)
            )
            selected = upper if np.random.rand() < upper_probability else lower
        points = keypoints[selected]
        center = points.mean(axis=0).astype(np.float32)
        extent = points.max(axis=0) - points.min(axis=0)
        half_body_scale = float(self.augmentation_config.get("half_body_scale_factor", 1.5))
        scale = np.maximum(extent * half_body_scale, 1.0).astype(np.float32)
        return center, scale

    def _augment_bbox(self, center, scale):
        random_values = truncnorm.rvs(-1.0, 1.0, size=(4,)).astype(np.float32)
        if np.random.rand() < float(self.augmentation_config.get("bbox_shift_probability", 0.3)):
            center = center + random_values[:2] * float(
                self.augmentation_config.get("bbox_shift_factor", 0.16)
            ) * scale
        scale_factor = random_values[2] * float(
            self.augmentation_config.get("bbox_scale_factor", 0.5)
        ) + 1.0
        scale = scale * scale_factor
        rotation = (
            float(random_values[3] * float(
                self.augmentation_config.get("bbox_rotation_factor", 80.0)
            ))
            if np.random.rand() < float(
                self.augmentation_config.get("bbox_rotation_probability", 0.6)
            ) else 0.0
        )
        return center, scale, rotation

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self._entries[int(index)]
        image = self._read_image(entry["image_path"])
        height, width = image.shape[:2]
        bbox = entry["bbox"].copy()
        keypoints = entry["keypoints"].copy()
        visible = entry["visible"].copy()

        center = (bbox[:2] + bbox[2:]) * 0.5
        scale = (bbox[2:] - bbox[:2]) * float(
            self.preprocessing_config.get("bbox_padding", 1.25)
        )
        rotation = 0.0
        if self.training:
            if self.flip_indices is not None and np.random.rand() < float(
                self.augmentation_config.get("horizontal_flip_probability", 0.5)
            ):
                image = np.ascontiguousarray(image[:, ::-1])
                center[0] = width - 1 - center[0]
                keypoints[:, 0] = width - 1 - keypoints[:, 0]
                keypoints = keypoints[self.flip_indices]
                visible = visible[self.flip_indices]
            center, scale = self._apply_half_body(center, scale, keypoints, visible)
            center, scale, rotation = self._augment_bbox(center, scale)

        scale = _fix_aspect_ratio(scale, self.input_size)
        warp_matrix = _topdown_warp_matrix(center, scale, rotation, self.input_size)
        inverse_matrix = cv2.invertAffineTransform(warp_matrix).astype(np.float32)
        crop = cv2.warpAffine(
            image,
            warp_matrix,
            self.input_size,
            flags=cv2.INTER_LINEAR,
        )
        transformed_points = cv2.transform(keypoints[None, :, :], warp_matrix)[0]
        labels_x, labels_y, target_weights = self.codec.encode(transformed_points, visible)

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        mean = torch.as_tensor(
            self.preprocessing_config.get("pixel_mean", self.PIXEL_MEAN),
            dtype=image_tensor.dtype,
        ).view(3, 1, 1)
        std = torch.as_tensor(
            self.preprocessing_config.get("pixel_std", self.PIXEL_STD),
            dtype=image_tensor.dtype,
        ).view(3, 1, 1)
        image_tensor.sub_(mean).div_(std)

        return {
            "images": image_tensor,
            "targets": {
                "simcc_x": torch.from_numpy(labels_x),
                "simcc_y": torch.from_numpy(labels_y),
                "keypoint_weights": torch.from_numpy(target_weights),
                "keypoints": torch.from_numpy(entry["keypoints"].copy()),
                "keypoints_visible": torch.from_numpy(entry["visible"].copy()),
                "bbox_xyxy": torch.from_numpy(entry["bbox"].copy()),
                "warp_inverse": torch.from_numpy(inverse_matrix),
                "image_id": int(entry["image_id"]),
                "category_id": int(entry["category_id"]),
                "annotation_id": int(entry["annotation_id"]),
                "area": float(entry["area"]),
                "image_path": entry["image_path"],
            },
        }


def build_topdown_dataloaders(
    data_yaml: Union[str, Path],
    input_size: Tuple[int, int],
    batch_size: int,
    validation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 0,
    include_validation: bool = True,
    cache: Union[bool, str] = False,
    fraction: float = 1.0,
    sigma: float = 6.0,
    split_ratio: float = 2.0,
    augmentation_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Build MMPose-style instance-crop loaders for the shared training engine."""
    from .engine import _require_torch

    torch = _require_torch()
    config_path = Path(data_yaml).expanduser().resolve()
    train_dataset = TopDownPoseDataset(
        config_path,
        "train",
        input_size=input_size,
        sigma=sigma,
        split_ratio=split_ratio,
        cache=cache,
        fraction=fraction,
        seed=seed,
        augmentation_config=augmentation_config,
        preprocessing_config=preprocessing_config,
    )
    values = _read_yaml(config_path)
    validation_value = values.get("val") or values.get("val_annotations")
    if validation_value is None and isinstance(values.get("annotations"), dict):
        validation_value = values["annotations"].get("val")
    validation_dataset = None
    if include_validation and validation_value is not None:
        validation_dataset = TopDownPoseDataset(
            config_path,
            "val",
            input_size=input_size,
            sigma=sigma,
            split_ratio=split_ratio,
            cache=cache,
            seed=seed,
            augmentation_config=augmentation_config,
            preprocessing_config=preprocessing_config,
        )

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=int(seed))
        if distributed else None
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed) + rank)
    loader_options = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": int(num_workers) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **loader_options,
    )
    validation_loader = None
    if validation_dataset is not None:
        # MMPose's validation sampler does not pad/duplicate the validation set. Each rank
        # evaluates the full set so the shared Trainer can keep validation aggregation simple.
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(validation_batch_size or batch_size),
            sampler=SequentialSampler(validation_dataset),
            **loader_options,
        )

    metadata_dataset = validation_dataset or train_dataset
    metadata = {
        "train_images": len(train_dataset),
        "train_instances": len(train_dataset),
        "validation_instances": len(validation_dataset) if validation_dataset is not None else 0,
        "num_classes": 1,
        "class_names": ["animal"],
        "kpt_shape": [metadata_dataset.num_keypoints, metadata_dataset.keypoint_dimensions],
        "kpt_oks_sigmas": metadata_dataset.kpt_oks_sigmas.tolist(),
        "flip_idx": (
            metadata_dataset.flip_indices.tolist()
            if metadata_dataset.flip_indices is not None else None
        ),
        "annotation_path": str(metadata_dataset.annotation_path)
        if metadata_dataset.annotation_path is not None else None,
        "annotation_format": metadata_dataset.annotation_format,
        "keypoint_names": metadata_dataset.keypoint_names,
        "coco_categories": metadata_dataset.annotations._coco_category_to_class
        if metadata_dataset.annotations._coco_category_to_class is not None else {},
    }
    return train_loader, validation_loader, metadata


class SimCCPoseValidator:
    """Compute MMPose-compatible SimCC decoding and keypoint metrics."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        split_ratio: float,
        flip_indices: Optional[Sequence[int]],
        kpt_oks_sigmas: Sequence[float],
        validation_dataset,
        coco_max_detections: int = 20,
        oks_nms_threshold: float = 0.9,
        keypoint_score_threshold: float = 0.2,
        pck_threshold: float = 0.05,
        auc_norm_factor: float = 30.0,
        auc_thresholds: int = 20,
    ) -> None:
        self.input_size = tuple(map(int, input_size))
        self.split_ratio = float(split_ratio)
        self.flip_indices = (
            torch.as_tensor(flip_indices, dtype=torch.long) if flip_indices is not None else None
        )
        self.kpt_oks_sigmas = np.asarray(kpt_oks_sigmas, dtype=np.float32)
        self.validation_dataset = validation_dataset
        self.coco_max_detections = int(coco_max_detections)
        self.oks_nms_threshold = float(oks_nms_threshold)
        self.keypoint_score_threshold = float(keypoint_score_threshold)
        self.pck_threshold = float(pck_threshold)
        self.auc_norm_factor = float(auc_norm_factor)
        self.auc_thresholds = int(auc_thresholds)

    def __call__(self, model, loader, device):
        predictions_for_coco = []
        errors = []
        visibility_parts = []
        bbox_sizes = []
        if self.validation_dataset is None:
            raise ValueError("AnimalViTPose validation requires the validation dataset")
        coco_gt, _image_ids_by_path, _category_ids_by_class = build_coco_ground_truth(
            self.validation_dataset.annotations
        )

        with torch.inference_mode():
            for batch in loader:
                images = batch["images"].to(device, non_blocking=device.type == "cuda")
                targets = batch["targets"]
                pred_x, pred_y = model(images)
                if self.flip_indices is not None:
                    flipped_x, flipped_y = model(torch.flip(images, dims=(-1,)))
                    permutation = self.flip_indices.to(pred_x.device)
                    flipped_x = flipped_x.flip(dims=(-1,)).index_select(1, permutation)
                    flipped_y = flipped_y.index_select(1, permutation)
                    pred_x = (pred_x + flipped_x) * 0.5
                    pred_y = (pred_y + flipped_y) * 0.5

                max_x, coords_x = pred_x.max(dim=-1)
                max_y, coords_y = pred_y.max(dim=-1)
                keypoint_scores = torch.minimum(max_x, max_y)
                crop_coords = torch.stack((coords_x, coords_y), dim=-1).float()
                crop_coords /= self.split_ratio
                invalid = keypoint_scores <= 0
                crop_coords[invalid] = -1.0 / self.split_ratio
                crop_coords = crop_coords.detach().cpu().numpy()
                keypoint_scores = keypoint_scores.detach().cpu().numpy()
                inverse = targets["warp_inverse"].numpy()
                homogeneous = np.concatenate(
                    [crop_coords, np.ones((*crop_coords.shape[:2], 1), dtype=np.float32)], axis=-1
                )
                source_coords = np.einsum("bij,bkj->bki", inverse, homogeneous)

                gt_coords = targets["keypoints"].numpy()
                visible = targets["keypoints_visible"].numpy() > 0
                boxes = targets["bbox_xyxy"].numpy()
                ids = targets["image_id"].numpy().tolist()
                categories = targets["category_id"].numpy().tolist()
                areas = targets["area"].numpy().tolist()

                for sample_index in range(len(source_coords)):
                    sample_visible = visible[sample_index]
                    delta = source_coords[sample_index] - gt_coords[sample_index]
                    distance = np.linalg.norm(delta, axis=-1)
                    errors.append(distance.astype(np.float32))
                    visibility_parts.append(sample_visible)
                    bbox = boxes[sample_index]
                    bbox_sizes.append(max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])))

                    if coco_gt is not None:
                        sample_scores = keypoint_scores[sample_index]
                        keypoint_array = np.concatenate(
                            [source_coords[sample_index], sample_scores[:, None]], axis=1
                        )
                        valid_scores = sample_scores[
                            sample_scores > self.keypoint_score_threshold
                        ]
                        confidence = float(valid_scores.mean()) if valid_scores.size else 0.0
                        predictions_for_coco.append({
                            "image_id": int(ids[sample_index]),
                            "category_id": int(categories[sample_index]),
                            "keypoints": keypoint_array.reshape(-1).tolist(),
                            "score": confidence,
                            "area": float(areas[sample_index]),
                        })

        if not errors:
            raise ValueError("Validation data loader produced no top-down samples")
        errors = np.stack(errors)
        visible = np.stack(visibility_parts)
        bbox_sizes = np.maximum(np.asarray(bbox_sizes, dtype=np.float32), 1.0)
        metrics = keypoint_error_metrics(
            errors,
            visible,
            bbox_sizes,
            pck_threshold=self.pck_threshold,
            auc_norm_factor=self.auc_norm_factor,
            auc_thresholds=self.auc_thresholds,
        )
        metrics.update(self._coco_metrics(coco_gt, predictions_for_coco))
        return metrics

    def _coco_metrics(self, coco_gt, detections):
        detections = self._oks_nms(detections, threshold=self.oks_nms_threshold)
        return evaluate_coco_keypoints(
            coco_gt,
            detections,
            self.kpt_oks_sigmas,
            max_detections=self.coco_max_detections,
        )

    def _oks_nms(self, detections, threshold: float):
        """Match MMPose CocoMetric's default hard OKS NMS for top-down predictions."""
        grouped = {}
        for detection in detections:
            group_key = (
                int(detection["image_id"]),
                int(detection.get("category_id", 1)),
            )
            grouped.setdefault(group_key, []).append(detection)
        kept = []
        variances = (self.kpt_oks_sigmas * 2.0) ** 2
        for instances in grouped.values():
            order = sorted(
                range(len(instances)),
                key=lambda index: instances[index]["score"],
                reverse=True,
            )
            image_kept = []
            while order:
                selected_index = order[0]
                selected = instances[selected_index]
                image_kept.append(selected)
                selected_keypoints = np.asarray(selected["keypoints"], dtype=np.float32).reshape(-1, 3)
                remaining = []
                for candidate_index in order[1:]:
                    candidate = instances[candidate_index]
                    candidate_keypoints = np.asarray(candidate["keypoints"], dtype=np.float32).reshape(-1, 3)
                    delta = candidate_keypoints[:, :2] - selected_keypoints[:, :2]
                    denominator = (
                        variances
                        * ((float(selected["area"]) + float(candidate["area"])) * 0.5 + np.spacing(1))
                        * 2.0
                    )
                    oks = float(np.exp(-np.sum(delta * delta, axis=1) / denominator).mean())
                    if oks <= threshold:
                        remaining.append(candidate_index)
                order = remaining
            kept.extend(image_kept)
        return kept


__all__ = [
    "SimCCLabel",
    "SimCCKLLoss",
    "SimCCPoseValidator",
    "TopDownPoseDataset",
    "build_topdown_dataloaders",
]
