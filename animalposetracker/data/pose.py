"""AnimalPoseTracker pose-data reader for existing project image/label folders."""

import hashlib
import multiprocessing
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .augment import PoseAugment
from .coco import annotation_source_for_split, load_coco_pose_index


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Dataset configuration must be a YAML mapping: {path}")
    return values


def _resolve_root(data_config: Dict[str, Any], config_path: Path) -> Path:
    configured = data_config.get("path")
    root = Path(configured).expanduser() if configured else config_path.parent
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def _resolve_config_path(value: Any, root: Path, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    rooted = (root / path).resolve()
    return rooted if rooted.exists() else (config_path.parent / path).resolve()


def _expand_sources(value: Any, root: Path, split: str) -> List[Path]:
    if value is None:
        return []
    entries = value if isinstance(value, (list, tuple)) else [value]
    images: List[Path] = []
    for entry in entries:
        path = Path(str(entry)).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_dir():
            images.extend(
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        elif path.is_file() and path.suffix.lower() in {".txt", ".list"}:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                candidate_text = line.strip()
                if not candidate_text:
                    continue
                candidate = Path(candidate_text).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                    if not candidate.exists():
                        candidate = root / candidate_text
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(candidate.resolve())
                else:
                    raise FileNotFoundError(
                        f"Image listed in the {split} split was not found: {candidate}"
                    )
        else:
            raise FileNotFoundError(f"{split} image source does not exist: {path}")
    return sorted(set(images))


def _label_path(image_path: Path) -> Path:
    """Map the established images/<split>/ path to labels/<split>/ by relative path."""
    parts = list(image_path.parts)
    image_indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, float(left), float(top)


class PoseTextDataset(Dataset):
    """Read YOLO Pose TXT or COCO Keypoints JSON into normalized image batches.

    YOLO labels use ``class, cx, cy, width, height, keypoints...`` with normalized
    coordinates. COCO annotations use the standard ``images/annotations/categories``
    JSON records and pixel-space xywh boxes/keypoints. Neither path imports a trainer.
    """

    def __init__(
        self,
        data_yaml: Union[str, Path],
        split: str = "train",
        image_size: int = 640,
        cache: Union[bool, str] = False,
        fraction: float = 1.0,
        seed: int = 0,
        augmentation: Optional[Dict[str, Any]] = None,
        epochs: int = 100,
        close_mosaic: int = 10,
        single_cls: bool = False,
    ) -> None:
        self.config_path = Path(data_yaml).expanduser().resolve()
        self.data_config = _read_yaml(self.config_path)
        self.root = _resolve_root(self.data_config, self.config_path)
        self.split = split
        self.image_size = int(image_size)
        self.cache_mode = "ram" if cache is True else str(cache).strip().lower() if cache else "none"
        self._ram_cache = {}
        self._epoch = multiprocessing.Value("i", 0, lock=False)
        self.epochs = int(epochs)
        self.close_mosaic = int(close_mosaic)
        self.single_cls = bool(single_cls)
        self.augmentation = dict(augmentation or {}) if split == "train" else {}
        if self.image_size < 1:
            raise ValueError("image_size must be at least 1")
        if not 0 < float(fraction) <= 1:
            raise ValueError("fraction must be in the range (0, 1]")

        sources = self.data_config.get(split)
        annotation_source = annotation_source_for_split(self.data_config, split)
        source_is_json = (
            isinstance(sources, (str, Path)) and Path(str(sources)).suffix.lower() == ".json"
        )
        if annotation_source is None and source_is_json:
            annotation_source = sources
        annotation_format = str(
            self.data_config.get("annotation_format", self.data_config.get("format", "auto"))
        ).strip().lower()
        if annotation_format not in {"auto", "yolo", "coco"}:
            raise ValueError("annotation_format must be 'auto', 'yolo' or 'coco'")
        if annotation_format == "coco" and annotation_source is None:
            candidates = (
                self.root / "annotations" / f"person_keypoints_{split}2017.json",
                self.root / "annotations" / f"{split}.json",
                self.root / f"{split}.json",
            )
            annotation_source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if annotation_source is None:
                raise FileNotFoundError(
                    f"COCO format was selected but no {split} annotation JSON was configured or found"
                )

        names = self.data_config.get("names") or {}
        class_names = list(names.values()) if isinstance(names, dict) else list(names)
        self.annotation_path = None
        self._coco_annotations_by_path = None
        self._coco_image_metadata_by_path = {}
        self._coco_category_to_class = {}
        if annotation_source is not None:
            if annotation_format == "yolo":
                raise ValueError("A COCO JSON annotation source conflicts with annotation_format='yolo'")
            self.annotation_path = _resolve_config_path(annotation_source, self.root, self.config_path)
            if not self.annotation_path.is_file():
                raise FileNotFoundError(f"COCO annotation JSON does not exist: {self.annotation_path}")

            image_source = self.data_config.get(f"{split}_images")
            if image_source is None and isinstance(self.data_config.get("images"), dict):
                image_source = self.data_config["images"].get(split)
            if image_source is None and not source_is_json:
                image_source = sources
            available_images = (
                _expand_sources(image_source, self.root, split) if image_source is not None else []
            )
            image_roots = sorted({path.parent for path in available_images})
            if image_source is None and source_is_json:
                inferred_roots = (
                    self.root / "images",
                    self.root / "images" / f"{split}2017",
                    self.root / f"{split}2017",
                    self.root / "images" / split,
                    self.root / split,
                )
                image_roots.extend(path.resolve() for path in inferred_roots if path.is_dir())
            (
                self.image_paths,
                self._coco_annotations_by_path,
                class_names,
                self._coco_category_to_class,
                inferred_shape,
                self._coco_image_metadata_by_path,
            ) = load_coco_pose_index(
                self.annotation_path,
                self.root,
                image_roots,
                available_images,
                class_names or None,
                self.data_config.get("kpt_shape"),
            )
            shape = self.data_config.get("kpt_shape") or inferred_shape
        else:
            if annotation_format == "coco":
                raise ValueError("COCO format requires a COCO annotation JSON")
            shape = self.data_config.get("kpt_shape")
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(
                    f"Dataset configuration needs kpt_shape=[count, dimensions]: {self.config_path}"
                )
            self.image_paths = _expand_sources(sources, self.root, split)
            if not class_names:
                raise ValueError(f"Dataset configuration needs class names: {self.config_path}")

        self.num_keypoints, self.keypoint_dimensions = map(int, shape)
        if self.num_keypoints < 1 or self.keypoint_dimensions not in (2, 3):
            raise ValueError(f"Unsupported keypoint shape {shape!r} in {self.config_path}")
        unsupported_sigma_fields = [
            name for name in ("oks_sigmas", "sigmas") if name in self.data_config
        ]
        if unsupported_sigma_fields:
            raise ValueError(
                "Dataset YAML uses unsupported OKS sigma field(s) "
                f"{unsupported_sigma_fields}; rename the vector to kpt_oks_sigmas"
            )
        self.kpt_oks_sigmas = self.data_config.get("kpt_oks_sigmas")
        if self.kpt_oks_sigmas is None:
            self.kpt_oks_sigmas = [1.0 / self.num_keypoints] * self.num_keypoints
            warnings.warn(
                f"No per-keypoint OKS sigma vector is configured for {self.config_path}; "
                f"using the uniform 1/{self.num_keypoints} custom-keypoint fallback. "
                "Set kpt_oks_sigmas in the dataset YAML for "
                "dataset-specific animal keypoint tolerances.",
                UserWarning,
                stacklevel=2,
            )
        self.kpt_oks_sigmas = np.asarray(
            self.kpt_oks_sigmas, dtype=np.float32
        ).reshape(-1).tolist()
        if len(self.kpt_oks_sigmas) != self.num_keypoints or not np.isfinite(
            self.kpt_oks_sigmas
        ).all() or any(
            sigma <= 0 for sigma in self.kpt_oks_sigmas
        ):
            raise ValueError(
                f"Per-keypoint OKS sigmas must contain {self.num_keypoints} finite positive values"
            )

        self.source_class_names = list(class_names)
        if not self.source_class_names:
            raise ValueError(f"Dataset configuration needs at least one class name: {self.config_path}")
        self.class_names = ["item"] if self.single_cls else self.source_class_names
        self.skeleton = self.data_config.get("skeleton") or []
        self.flip_idx = self.data_config.get("flip_idx")
        if self.flip_idx is not None and len(self.flip_idx) != self.num_keypoints:
            raise ValueError("flip_idx length must equal the number of keypoints")
        if self.flip_idx is not None and sorted(map(int, self.flip_idx)) != list(range(self.num_keypoints)):
            raise ValueError("flip_idx must be a permutation of keypoint indices")

        if not self.image_paths:
            raise ValueError(f"The {split} split contains no images: {self.config_path}")
        self.image_id_by_path = {
            str(path.resolve()): int(self._coco_image_metadata_by_path[str(path.resolve())]["id"])
            for path in self.image_paths
            if str(path.resolve()) in self._coco_image_metadata_by_path
        }
        if not self.image_id_by_path:
            self.image_id_by_path = {
                str(path.resolve()): index + 1
                for index, path in enumerate(self.image_paths)
            }
        self.category_id_by_class = (
            {
                int(class_id): int(category_id)
                for category_id, class_id in self._coco_category_to_class.items()
            }
            if self._coco_category_to_class else
            {class_id: class_id + 1 for class_id in range(len(self.source_class_names))}
        )
        if self.flip_idx is None and (
            float(self.augmentation.get("fliplr", 0.0)) > 0
            or float(self.augmentation.get("flipud", 0.0)) > 0
        ):
            warnings.warn(
                "Dataset has no flip_idx mapping; horizontal and vertical flips are disabled "
                "for keypoint-label consistency.",
                UserWarning,
                stacklevel=2,
            )
            self.augmentation["fliplr"] = 0.0
            self.augmentation["flipud"] = 0.0
        self.augmenter = (
            PoseAugment(self.image_size, self.augmentation, self.flip_idx)
            if self.augmentation else None
        )
        self.total_image_count = len(self.image_paths)
        if split == "train" and fraction < 1.0:
            count = max(1, int(round(len(self.image_paths) * float(fraction))))
            rng = np.random.default_rng(int(seed))
            chosen = np.sort(rng.choice(len(self.image_paths), size=count, replace=False))
            self.image_paths = [self.image_paths[int(index)] for index in chosen]
        if self.cache_mode == "disk":
            self.disk_cache_dir = Path(tempfile.gettempdir()) / "AnimalPoseTracker" / "cache" / hashlib.sha1(
                f"{self.config_path}|{self.split}|{self.image_size}".encode("utf-8")
            ).hexdigest()[:16]
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        elif self.cache_mode not in ("none", "ram"):
            raise ValueError("cache must be false, true, 'ram' or 'disk'")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._load_base_item(index)
        if self.augmenter is not None:
            sample = self.augmenter.apply(
                sample,
                self._load_base_item,
                len(self.image_paths),
                allow_mosaic=self._mosaic_enabled(),
            )
            image, classes, boxes, keypoints = (
                sample["image"], sample["classes"], sample["boxes"], sample["keypoints"]
            )
            image_paths = sample["paths"]
        else:
            image, classes, boxes, keypoints = (
                sample["image"], sample["classes"], sample["boxes"], sample["keypoints"]
            )
            image_paths = sample["path"]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)

        return {
            "images": image_tensor,
            "targets": {
                "classes": torch.as_tensor(classes, dtype=torch.long),
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "keypoints": torch.as_tensor(keypoints, dtype=torch.float32).reshape(
                    -1, self.num_keypoints, self.keypoint_dimensions
                ),
                "image_paths": image_paths,
            },
        }

    def set_epoch(self, epoch: int) -> None:
        """Share the current epoch with persistent data-loader workers."""
        self._epoch.value = int(epoch)

    def _mosaic_enabled(self) -> bool:
        if self.close_mosaic <= 0 or self.epochs <= 0:
            return True
        cutoff = self.epochs - self.close_mosaic
        if cutoff < 0:
            return True
        return int(self._epoch.value) < cutoff

    def _cache_path(self, index: int) -> Path:
        image_path = self.image_paths[index]
        label_path = _label_path(image_path)
        image_stat = image_path.stat()
        label_stat = label_path.stat() if label_path.is_file() else None
        annotation_stat = self.annotation_path.stat() if self.annotation_path is not None else None
        fingerprint = (
            f"{image_path}|{image_stat.st_size}|{image_stat.st_mtime_ns}|"
            f"{label_path}|{label_stat.st_size if label_stat else 0}|"
            f"{label_stat.st_mtime_ns if label_stat else 0}|"
            f"{annotation_stat.st_size if annotation_stat else 0}|"
            f"{annotation_stat.st_mtime_ns if annotation_stat else 0}|{self.image_size}"
        )
        return self.disk_cache_dir / (hashlib.sha1(fingerprint.encode("utf-8")).hexdigest() + ".npz")

    def _load_base_item(self, index: int) -> Dict[str, np.ndarray]:
        index = int(index) % len(self.image_paths)
        if self.cache_mode == "ram" and index in self._ram_cache:
            cached = self._ram_cache[index]
            return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in cached.items()}
        cache_path = self._cache_path(index) if self.cache_mode == "disk" else None
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    sample = {
                        "image": cached["image"].copy(),
                        "classes": cached["classes"].copy(),
                        "boxes": cached["boxes"].copy(),
                        "keypoints": cached["keypoints"].copy(),
                        "path": str(self.image_paths[index]),
                    }
                return sample
            except (OSError, ValueError, KeyError):
                cache_path.unlink(missing_ok=True)

        path = self.image_paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        source_height, source_width = image.shape[:2]
        image, scale, pad_x, pad_y = _letterbox(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classes, boxes, keypoints = self._read_targets(path, source_width, source_height)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] = (boxes[:, 0] * source_width * scale + pad_x) / self.image_size
            boxes[:, 1] = (boxes[:, 1] * source_height * scale + pad_y) / self.image_size
            boxes[:, 2] *= source_width * scale / self.image_size
            boxes[:, 3] *= source_height * scale / self.image_size
            keypoints = keypoints.copy()
            keypoints[:, :, 0] = (keypoints[:, :, 0] * source_width * scale + pad_x) / self.image_size
            keypoints[:, :, 1] = (keypoints[:, :, 1] * source_height * scale + pad_y) / self.image_size
        sample = {"image": image, "classes": classes, "boxes": boxes, "keypoints": keypoints, "path": str(path)}
        if self.cache_mode == "ram":
            self._ram_cache[index] = sample
        elif cache_path is not None:
            temporary = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
            np.savez(temporary, image=image, classes=classes, boxes=boxes, keypoints=keypoints)
            try:
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in sample.items()}

    def _read_targets(self, image_path: Path, image_width: int, image_height: int):
        if self._coco_annotations_by_path is not None:
            return self._read_coco_targets(image_path, image_width, image_height)
        label_path = _label_path(image_path)
        if not label_path.is_file():
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, self.num_keypoints, self.keypoint_dimensions), dtype=np.float32),
            )

        classes: List[int] = []
        boxes: List[List[float]] = []
        keypoints: List[List[List[float]]] = []
        expected_keypoints = self.num_keypoints * self.keypoint_dimensions
        with label_path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) not in (5, 5 + expected_keypoints):
                    raise ValueError(
                        f"Expected 5 or {5 + expected_keypoints} values at "
                        f"{label_path}:{line_number}, got {len(fields)}"
                    )
                try:
                    class_value = float(fields[0])
                    box = [float(value) for value in fields[1:5]]
                    values = [float(value) for value in fields[5:]]
                except ValueError as exc:
                    raise ValueError(f"Non-numeric annotation at {label_path}:{line_number}") from exc
                if not np.isfinite([class_value, *box, *values]).all():
                    raise ValueError(f"Non-finite annotation at {label_path}:{line_number}")
                class_id = int(class_value)
                if class_id != class_value or class_id < 0 or class_id >= len(self.source_class_names):
                    raise ValueError(f"Class id {fields[0]} is outside the configured class range at {label_path}:{line_number}")
                if box[2] <= 0 or box[3] <= 0:
                    raise ValueError(f"Bounding-box width and height must be positive at {label_path}:{line_number}")

                points = np.zeros(
                    (self.num_keypoints, self.keypoint_dimensions), dtype=np.float32
                )
                if values:
                    points = np.asarray(values, dtype=np.float32).reshape(
                        self.num_keypoints, self.keypoint_dimensions
                    )
                elif self.keypoint_dimensions == 3:
                    # A box-only label has no supervised keypoints.
                    points[:, 2] = 0.0
                else:
                    raise ValueError(
                        f"A box-only annotation cannot be used with kpt_shape=[{self.num_keypoints}, 2] "
                        f"because this format has no visibility mask: {label_path}:{line_number}"
                    )
                classes.append(0 if self.single_cls else class_id)
                boxes.append(box)
                keypoints.append(points.tolist())

        return (
            np.asarray(classes, dtype=np.int64),
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(keypoints, dtype=np.float32).reshape(
                -1, self.num_keypoints, self.keypoint_dimensions
            ),
        )

    def _read_coco_targets(self, image_path: Path, image_width: int, image_height: int):
        classes: List[int] = []
        boxes: List[List[float]] = []
        keypoints: List[List[List[float]]] = []
        records = self._coco_annotations_by_path.get(str(image_path.resolve()), [])
        for record in records:
            class_id = self._coco_category_to_class[int(record["category_id"])]
            bbox = np.asarray(record.get("bbox", ()), dtype=np.float32)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            x1 = float(np.clip(bbox[0], 0, image_width))
            y1 = float(np.clip(bbox[1], 0, image_height))
            x2 = float(np.clip(bbox[0] + bbox[2], 0, image_width))
            y2 = float(np.clip(bbox[1] + bbox[3], 0, image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([
                ((x1 + x2) * 0.5) / image_width,
                ((y1 + y2) * 0.5) / image_height,
                (x2 - x1) / image_width,
                (y2 - y1) / image_height,
            ])

            points = np.zeros(
                (self.num_keypoints, self.keypoint_dimensions), dtype=np.float32
            )
            values = record.get("keypoints") or []
            if values:
                if len(values) != self.num_keypoints * 3:
                    raise ValueError(
                        f"Expected {self.num_keypoints * 3} COCO keypoint values for "
                        f"image {image_path}, got {len(values)}"
                    )
                points3 = np.asarray(values, dtype=np.float32).reshape(self.num_keypoints, 3)
                if not np.isfinite(points3).all():
                    raise ValueError(f"Non-finite COCO keypoint annotation for image {image_path}")
                if self.keypoint_dimensions == 2 and np.any(points3[:, 2] == 0):
                    raise ValueError(
                        f"COCO annotation {record.get('id', '<unknown>')} contains unlabeled "
                        "keypoints, but kpt_shape has no visibility channel"
                    )
                points[:, :2] = points3[:, :2] / np.asarray(
                    [image_width, image_height], dtype=np.float32
                )
                if self.keypoint_dimensions == 3:
                    points[:, 2] = points3[:, 2]
            elif self.keypoint_dimensions == 3:
                points[:, 2] = 0.0
            else:
                raise ValueError(
                    f"COCO annotation {record.get('id', '<unknown>')} has no keypoints, but "
                    f"kpt_shape=[{self.num_keypoints}, 2] has no visibility mask"
                )
            classes.append(0 if self.single_cls else class_id)
            keypoints.append(points.tolist())

        return (
            np.asarray(classes, dtype=np.int64),
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(keypoints, dtype=np.float32).reshape(
                -1, self.num_keypoints, self.keypoint_dimensions
            ),
        )


def pose_collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack images and flatten variable-length object annotations with batch indices."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    images = torch.stack([sample["images"] for sample in samples], dim=0)
    target_parts = [sample["targets"] for sample in samples]
    batch_indices = [
        torch.full((part["boxes"].shape[0],), index, dtype=torch.long)
        for index, part in enumerate(target_parts)
    ]
    targets = {
        "batch_indices": torch.cat(batch_indices, dim=0),
        "classes": torch.cat([part["classes"] for part in target_parts], dim=0),
        "boxes": torch.cat([part["boxes"] for part in target_parts], dim=0),
        "keypoints": torch.cat([part["keypoints"] for part in target_parts], dim=0),
        "image_paths": [part["image_paths"] for part in target_parts],
    }
    return {"images": images, "targets": targets}


def build_pose_dataloaders(
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
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Construct project-owned training and validation loaders from dataset YAML."""
    config_path = Path(data_yaml).expanduser().resolve()
    train_dataset = PoseTextDataset(
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
    )
    data_config = _read_yaml(config_path)
    validation_value = data_config.get("val") or data_config.get("val_annotations")
    if validation_value is None and isinstance(data_config.get("annotations"), dict):
        validation_value = data_config["annotations"].get("val")
    validation_dataset = (
        PoseTextDataset(
            config_path, "val", image_size, cache=cache, single_cls=single_cls
        ) if validation_value else None
    ) if include_validation else None
    generator = torch.Generator()
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    generator.manual_seed(int(seed) + rank)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )
        if distributed else None
    )
    shared = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "collate_fn": pose_collate,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **shared,
    )
    validation_options = dict(shared)
    validation_options["batch_size"] = int(validation_batch_size or batch_size)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **validation_options)
        if validation_dataset is not None else None
    )
    metadata = {
        "num_classes": len(train_dataset.class_names),
        "class_names": train_dataset.class_names,
        "kpt_shape": [train_dataset.num_keypoints, train_dataset.keypoint_dimensions],
        "kpt_oks_sigmas": train_dataset.kpt_oks_sigmas,
        "flip_idx": train_dataset.flip_idx,
        "skeleton": train_dataset.skeleton,
        "train_images": len(train_dataset),
        "train_images_before_fraction": train_dataset.total_image_count,
        "val_images": len(validation_dataset) if validation_dataset is not None else 0,
    }
    return train_loader, validation_loader, metadata
