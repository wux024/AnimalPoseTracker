"""Typed configuration for AnimalPoseTracker's training process."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


@dataclass
class TrainingConfig:
    """Settings shared by training jobs, independent of a model's loss function.

    Paths are resolved against ``project_dir``. ``from_mapping`` accepts the
    legacy project keys (``batch``, ``imgsz`` and ``project``) so existing
    project files can be migrated in memory without being rewritten.
    """

    project_dir: PathLike = Path(".")
    model: Optional[PathLike] = None
    data: Optional[PathLike] = None
    pretrained_weights: Optional[PathLike] = None
    output_dir: PathLike = Path("runs/train")

    epochs: int = 100
    time_hours: Optional[float] = None
    batch_size: int = 16
    validation_batch_size: Optional[int] = None
    image_size: int = 640
    num_workers: int = 0
    device: str = "auto"
    cache: Union[bool, str] = False
    fraction: float = 1.0
    profile: bool = False
    plots: bool = True
    save: bool = True
    verbose: bool = True
    single_cls: bool = False
    multi_scale: bool = False
    freeze: Optional[Union[int, list]] = None

    optimizer: str = "auto"
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    momentum: float = 0.9
    final_lr_factor: float = 0.01
    cosine_schedule: bool = False
    warmup_epochs: float = 3.0
    warmup_iters: Optional[int] = None
    warmup_start_factor: Optional[float] = None
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    nominal_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = None
    layer_decay_rate: Optional[float] = None
    lr_milestones: Optional[list] = None
    lr_gamma: float = 0.1
    ema: bool = True
    amp: bool = True
    seed: int = 0
    deterministic: bool = True

    validation_interval: int = 1
    validation_enabled: bool = True
    validation_confidence: float = 0.001
    validation_iou: float = 0.7
    max_detections: int = 300
    agnostic_nms: bool = False
    best_metric: str = "coco/AP"
    best_metric_mode: str = "max"
    early_stopping_patience: int = 300
    save_period: int = 0
    log_interval: int = 20
    resume_from: Optional[PathLike] = None
    train_dataset_size: Optional[int] = None
    resolved_optimizer: Optional[str] = None
    resolved_learning_rate: Optional[float] = None
    resolved_momentum: Optional[float] = None
    resolved_weight_decay: Optional[float] = None

    box_loss_weight: float = 7.5
    class_loss_weight: float = 0.5
    distribution_loss_weight: float = 1.5
    keypoint_loss_weight: float = 12.0
    visibility_loss_weight: float = 1.0

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    close_mosaic: int = 10

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).expanduser().resolve()
        self.model = self._resolve_optional_path(self.model)
        self.data = self._resolve_optional_path(self.data)
        self.pretrained_weights = self._resolve_optional_path(self.pretrained_weights)
        self.output_dir = self._resolve_path(self.output_dir)
        self.resume_from = self._resolve_optional_path(self.resume_from)
        if isinstance(self.cache, str):
            self.cache = self.cache.strip().lower()

        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.time_hours is not None and self.time_hours <= 0:
            raise ValueError("time must be greater than 0 hours when set")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.validation_batch_size is not None and self.validation_batch_size < 1:
            raise ValueError("validation_batch_size must be at least 1")
        if self.image_size < 1:
            raise ValueError("image_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.cache not in (False, True, "ram", "disk"):
            raise ValueError("cache must be false, true, 'ram' or 'disk'")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in the range (0, 1]")
        if isinstance(self.freeze, bool) or (
            self.freeze is not None
            and not isinstance(self.freeze, int)
            and not isinstance(self.freeze, (list, tuple))
        ):
            raise ValueError("freeze must be an integer layer count, a list of layer indices, or null")
        if isinstance(self.freeze, int) and self.freeze < 0:
            raise ValueError("freeze layer count cannot be negative")
        if isinstance(self.freeze, (list, tuple)) and any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.freeze
        ):
            raise ValueError("freeze indices must be non-negative integers")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.nominal_batch_size < 1:
            raise ValueError("nominal_batch_size must be at least 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if self.warmup_iters is not None and self.warmup_iters < 0:
            raise ValueError("warmup_iters cannot be negative")
        if self.warmup_start_factor is not None and not 0 <= self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be between 0 and 1")
        if self.warmup_momentum < 0 or self.warmup_momentum >= 1:
            raise ValueError("warmup_momentum must be in the range [0, 1)")
        if self.warmup_bias_lr < 0:
            raise ValueError("warmup_bias_lr cannot be negative")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be at least 1")
        if not 0 <= self.validation_confidence <= 1:
            raise ValueError("validation_confidence must be between 0 and 1")
        if not 0 <= self.validation_iou <= 1:
            raise ValueError("validation_iou must be between 0 and 1")
        if self.max_detections < 1:
            raise ValueError("max_detections must be at least 1")
        if self.close_mosaic < 0:
            raise ValueError("close_mosaic cannot be negative")
        if self.save_period < 0:
            raise ValueError("save_period cannot be negative")
        if self.log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.momentum < 0 or self.momentum >= 1:
            raise ValueError("momentum must be in the range [0, 1)")
        if self.final_lr_factor < 0 or self.final_lr_factor > 1:
            raise ValueError("final_lr_factor must be in the range [0, 1]")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be greater than 0 when set")
        if self.layer_decay_rate is not None and not 0 < self.layer_decay_rate <= 1:
            raise ValueError("layer_decay_rate must be in the range (0, 1]")
        if self.lr_gamma <= 0 or self.lr_gamma > 1:
            raise ValueError("lr_gamma must be in the range (0, 1]")
        if self.lr_milestones is not None:
            milestones = [int(value) for value in self.lr_milestones]
            if any(value < 1 for value in milestones) or milestones != sorted(set(milestones)):
                raise ValueError("lr_milestones must be a strictly increasing list of positive epochs")
            self.lr_milestones = milestones
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.best_metric_mode not in ("min", "max"):
            raise ValueError("best_metric_mode must be 'min' or 'max'")
        for name in (
            "box_loss_weight",
            "class_loss_weight",
            "distribution_loss_weight",
            "keypoint_loss_weight",
            "visibility_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("flipud", "fliplr", "bgr", "mosaic", "mixup"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def _resolve_path(self, path: PathLike) -> Path:
        result = Path(path).expanduser()
        if not result.is_absolute():
            result = self.project_dir / result
        return result.resolve()

    def _resolve_optional_path(self, path: Optional[PathLike]) -> Optional[PathLike]:
        if path is None:
            return None
        if isinstance(path, str) and path.lower().startswith(("http://", "https://")):
            return path
        return self._resolve_path(path)

    @classmethod
    def from_mapping(
        cls,
        values: Dict[str, Any],
        project_dir: Optional[PathLike] = None,
    ) -> "TrainingConfig":
        """Build from a mapping, including an existing ``configs/other.yaml``."""
        values = dict(values or {})
        root = Path(project_dir or values.get("project_dir") or ".").expanduser().resolve()

        run_root = values.get("project", "runs")
        run_name = values.get("name", "train")
        output_dir = values.get("output_dir", values.get("save_dir"))
        if output_dir is None:
            output_dir = Path(run_root) / run_name

        batch_size = int(values.get("batch_size", values.get("batch", 16)))
        nominal_batch_size = int(values.get("nominal_batch_size", values.get("nbs", 64)))
        accumulation = values.get("gradient_accumulation_steps")
        if accumulation is None:
            accumulation = max(round(nominal_batch_size / batch_size), 1)

        pretrained = values.get("pretrained_weights")
        if pretrained is None and isinstance(values.get("pretrained"), str):
            pretrained = values["pretrained"]

        resume_from = values.get("resume_from")
        resume = values.get("resume", False)
        if resume_from is None and isinstance(resume, str):
            resume_from = resume
        elif resume_from is None and resume is True:
            resume_from = Path(output_dir) / "weights" / "last.pt"

        optimizer = values.get("optimizer", "auto")

        known = {
            "project_dir": root,
            "model": values.get("model"),
            "data": values.get("data"),
            "pretrained_weights": pretrained,
            "output_dir": output_dir,
            "epochs": values.get("epochs", 100),
            "time_hours": None if values.get("time") is None else float(values["time"]),
            "batch_size": batch_size,
            "validation_batch_size": (
                None if values.get("validation_batch_size") is None
                else int(values["validation_batch_size"])
            ),
            "image_size": values.get("image_size", values.get("imgsz", 640)),
            "num_workers": values.get("num_workers", values.get("workers", 0)),
            "device": (
                ",".join(map(str, values["device"]))
                if isinstance(values.get("device"), (list, tuple))
                else str(values.get("device") or "auto")
            ),
            "cache": values.get("cache", False),
            "fraction": float(values.get("fraction", 1.0)),
            "profile": bool(values.get("profile", False)),
            "plots": bool(values.get("plots", True)),
            "save": bool(values.get("save", True)),
            "verbose": bool(values.get("verbose", True)),
            "single_cls": bool(values.get("single_cls", False)),
            "multi_scale": bool(values.get("multi_scale", False)),
            "freeze": values.get("freeze"),
            "optimizer": str(optimizer),
            "learning_rate": values.get("learning_rate", values.get("lr0", 1e-2)),
            "weight_decay": values.get("weight_decay", 5e-4),
            "momentum": values.get("momentum", 0.9),
            "final_lr_factor": values.get("final_lr_factor", values.get("lrf", 0.01)),
            "cosine_schedule": values.get("cosine_schedule", values.get("cos_lr", False)),
            "warmup_epochs": values.get("warmup_epochs", 3.0),
            "warmup_iters": (
                None if values.get("warmup_iters") is None else int(values["warmup_iters"])
            ),
            "warmup_start_factor": (
                None if values.get("warmup_start_factor") is None
                else float(values["warmup_start_factor"])
            ),
            "warmup_momentum": values.get("warmup_momentum", 0.8),
            "warmup_bias_lr": values.get("warmup_bias_lr", 0.1),
            "nominal_batch_size": nominal_batch_size,
            "gradient_accumulation_steps": accumulation,
            "gradient_clip_norm": values.get("gradient_clip_norm"),
            "layer_decay_rate": (
                None if values.get("layer_decay_rate") is None
                else float(values["layer_decay_rate"])
            ),
            "lr_milestones": values.get("lr_milestones"),
            "lr_gamma": float(values.get("lr_gamma", 0.1)),
            "ema": bool(values.get("ema", True)),
            "amp": values.get("amp", True),
            "seed": values.get("seed", 0),
            "deterministic": values.get("deterministic", True),
            "validation_interval": values.get("validation_interval", 1),
            "validation_enabled": values.get("val", True),
            "validation_confidence": (
                0.001 if values.get("conf") is None else float(values.get("conf"))
            ),
            "validation_iou": float(values.get("iou", 0.7)),
            "max_detections": int(values.get("max_det", 300)),
            "agnostic_nms": values.get("agnostic_nms", False),
            "best_metric": values.get("best_metric", "coco/AP"),
            "best_metric_mode": values.get(
                "best_metric_mode",
                "min" if values.get("best_metric", "coco/AP") in {"loss", "val_loss"} else "max",
            ),
            "early_stopping_patience": values.get(
                "early_stopping_patience", values.get("patience", 300)
            ),
            "save_period": max(0, int(values.get("save_period", 0))),
            "log_interval": values.get("log_interval", 20),
            "resume_from": resume_from,
            "box_loss_weight": values.get("box_loss_weight", values.get("box", 7.5)),
            "class_loss_weight": values.get("class_loss_weight", values.get("cls", 0.5)),
            "distribution_loss_weight": values.get(
                "distribution_loss_weight", values.get("dfl", 1.5)
            ),
            "keypoint_loss_weight": values.get("keypoint_loss_weight", values.get("pose", 12.0)),
            "visibility_loss_weight": values.get(
                "visibility_loss_weight", values.get("kobj", 1.0)
            ),
            "hsv_h": float(values.get("hsv_h", 0.015)),
            "hsv_s": float(values.get("hsv_s", 0.7)),
            "hsv_v": float(values.get("hsv_v", 0.4)),
            "degrees": float(values.get("degrees", 0.0)),
            "translate": float(values.get("translate", 0.1)),
            "scale": float(values.get("scale", 0.5)),
            "shear": float(values.get("shear", 0.0)),
            "perspective": float(values.get("perspective", 0.0)),
            "flipud": float(values.get("flipud", 0.0)),
            "fliplr": float(values.get("fliplr", 0.5)),
            "bgr": float(values.get("bgr", 0.0)),
            "mosaic": float(values.get("mosaic", 1.0)),
            "mixup": float(values.get("mixup", 0.0)),
            "close_mosaic": int(values.get("close_mosaic", 10)),
        }
        return cls(**known)

    @classmethod
    def from_yaml(
        cls,
        path: PathLike,
        project_dir: Optional[PathLike] = None,
    ) -> "TrainingConfig":
        """Load a YAML configuration without changing the source project file."""
        import yaml

        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Training configuration must be a YAML mapping: {config_path}")
        return cls.from_mapping(values, project_dir=project_dir or config_path.parent.parent)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly copy of this configuration."""
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
