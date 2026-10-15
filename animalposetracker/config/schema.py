"""The configuration schema: one registry entry per configuration key.

This module is the single place where a configuration key is *defined* -- its
canonical name, which scope and group it lives in, its type, when it takes
effect and whether the GUI shows it. Default *values* live in the defaults
files (``cfg/training.yaml``, ``cfg/runtime_defaults.yaml``); this registry
describes the keys those files and the project's ``runtime.yaml`` may carry.

The registry is data, not code: the resolver, the CLI builders, the GUI and
the generated documentation all read it, so none of them can drift from the
others (a key renamed here is renamed everywhere).

Terminology (mirrors the W0 concept dictionary, 2026-10-14):

- ``scope``: which flow group the key belongs to -- ``train`` (training and
  its validation mode), ``common`` (shared across flows), ``predict``,
  ``export``. The ``infer`` pipeline deliberately has no scope: it reads no
  project files (see the refactor plan, section 2.3).
- ``group``: the sub-section inside a scope. Train keys live under
  ``shared.runtime`` / ``shared.training`` / ``models.training`` /
  ``models.loss`` / ``models.augmentation`` / ``models.validation`` /
  ``metrics``; the ``models.*`` groups nest under the active model's profile.
  Predict keys mostly sit at the top of the section; ``tracking.*`` and
  ``pose_filter.*`` are mapping blocks.
- ``apply``: when a change takes effect for a live run -- ``build`` (session
  must be rebuilt), ``live`` (read fresh every frame), ``run_end`` (applied
  when the run finishes).
- ``visibility``: where the GUI shows the key -- ``form`` (regular),
  ``advanced`` (advanced/collapsed), ``hidden`` (the GUI does not show it).
- ``views``: the configure forms the key appears in -- ``train`` / ``evaluate`` /
  ``infer`` / ``export``. The GUI's parameter trees are generated from these
  annotations; a key without views stays off the forms.
- ``legacy``: spellings that appear in pre-refactor files or older CLI flags.
  NOT used for compatibility reads (old files are rejected); used to turn a
  stray old spelling into a helpful error message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

__all__ = [
    "Field",
    "FIELDS",
    "SCOPES",
    "APPLY_MODES",
    "VISIBILITIES",
    "TRAIN_GROUPS",
    "fields",
    "field_for",
    "to_cli_name",
    "display_path",
    "find_by_legacy",
]


SCOPES: Tuple[str, ...] = ("train", "common", "predict", "export")
APPLY_MODES: Tuple[str, ...] = ("build", "live", "run_end")
VISIBILITIES: Tuple[str, ...] = ("form", "advanced", "hidden")

TRAIN_GROUPS: Tuple[str, ...] = (
    "shared.runtime",
    "shared.training",
    "models.training",
    "models.loss",
    "models.augmentation",
    "models.validation",
    "metrics",
)


@dataclass(frozen=True)
class Field:
    """One configuration key.

    ``type`` accepts a single type or a tuple of accepted types; ``None``
    (null) is always accepted in addition (a key that is simply not set).
    """

    key: str
    scope: str
    group: str
    type: Any
    note: str
    apply: str = "build"
    visibility: str = "form"
    legacy: Tuple[str, ...] = ()
    views: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.type, type):
            object.__setattr__(self, "type", (self.type,))

    def accepts(self, value: Any) -> bool:
        """Whether ``value`` is acceptable for this key (``None`` always is)."""
        if value is None:
            return True
        for expected in self.type:
            if expected is float and isinstance(value, bool):
                continue  # bool is an int in Python; a float key does not take it
            if expected is int and isinstance(value, bool):
                continue
            if isinstance(value, expected):
                return True
        return False

    @property
    def path(self) -> str:
        """Human-readable location, e.g. ``predict.pose_filter.method``."""
        return display_path(self)


def _f(
    key: str,
    scope: str,
    group: str,
    type: Any,
    note: str,
    apply: str = "build",
    visibility: str = "form",
    legacy: Sequence[str] = (),
    views: Sequence[str] = (),
) -> Field:
    return Field(key, scope, group, type, note, apply, visibility, tuple(legacy), tuple(views))


# ---------------------------------------------------------------------------
# train -- shared.runtime (runtime settings)
# ---------------------------------------------------------------------------

FIELDS: Tuple[Field, ...] = (
    _f("model", "train", "shared.runtime", str, "Model graph path (derived from the project layout, not persisted)", visibility="hidden", views=("train",)),
    _f("data", "train", "shared.runtime", str, "Dataset path (derived from the project layout, not persisted)", visibility="hidden", views=("train", "evaluate", "export")),
    _f("num_workers", "train", "shared.runtime", int, "Data loading worker count", visibility="form", legacy=("workers",), views=("train", "evaluate")),
    _f("time_hours", "train", "shared.runtime", float, "Training time cap in hours; overrides epochs when set", visibility="advanced", legacy=("time",), views=("train",)),
    _f("cache", "train", "shared.runtime", (bool, str), "Data cache: false / true (ram) / ram / disk", visibility="form", views=("train",)),
    _f("fraction", "train", "shared.runtime", float, "Fraction of the training set to use", visibility="form", views=("train", "export")),
    _f("profile", "train", "shared.runtime", bool, "Per-stage timing profile", visibility="advanced", views=("train", "evaluate")),
    _f("plots", "train", "shared.runtime", bool, "Save training curve plots", visibility="form", views=("train", "evaluate")),
    _f("validation_enabled", "train", "shared.runtime", bool, "Run validation during training", visibility="form", legacy=("val",), views=("train",)),
    _f("seed", "train", "shared.runtime", int, "Random seed", visibility="form"),
    _f("deterministic", "train", "shared.runtime", bool, "Deterministic algorithms", visibility="advanced"),
    _f("save", "train", "shared.runtime", bool, "Save checkpoints", visibility="form", views=("train", "infer")),
    _f("save_period", "train", "shared.runtime", int, "Save an epoch checkpoint every N epochs; 0 = off", visibility="advanced", views=("train",)),
    _f("pretrained", "train", "shared.runtime", bool, "Start from pretrained weights", visibility="form", views=("train",)),
    _f("pretrained_weights", "train", "shared.runtime", str, "Explicit pretrained weights path", visibility="advanced"),
    _f("verbose", "train", "shared.runtime", bool, "Verbose logging", visibility="advanced", views=("evaluate", "infer")),
    _f("resume", "train", "shared.runtime", (bool, str), "Resume from the latest checkpoint", visibility="form", views=("train",)),
    _f("resume_from", "train", "shared.runtime", str, "Resume from a specific full checkpoint", visibility="hidden"),
    _f("channels_last", "train", "shared.runtime", bool, "Explicit channels_last switch; null = auto", visibility="advanced"),
    _f("gradient_accumulation_steps", "train", "models.training", int, "Gradient accumulation steps", visibility="advanced"),
    _f("gradient_clip_norm", "train", "models.training", float, "Gradient clipping norm; null = off", visibility="advanced"),
    _f("lr_milestones", "train", "models.training", list, "MultiStep learning-rate milestones", visibility="advanced"),
    _f("lr_gamma", "train", "shared.training", float, "Learning-rate decay at milestones", visibility="advanced"),
    _f("validation_interval", "train", "models.training", int, "Run validation every N epochs", visibility="advanced"),
    _f("log_interval", "train", "models.training", int, "Log every N batches", visibility="advanced"),
    _f("best_metric", "train", "shared.training", str, "Best-checkpoint metric (coco/AP or ultralytics/AP)", visibility="advanced"),
    _f("best_metric_mode", "train", "shared.training", str, "Selection direction: max / min", visibility="advanced"),
    _f("early_stopping_patience", "train", "shared.training", int, "Early-stopping patience in epochs", visibility="form", legacy=("patience",), views=("train",)),

    # train -- shared.training
    _f("single_cls", "train", "shared.training", bool, "Train multi-class data as a single class", visibility="advanced", views=("evaluate",)),
    _f("multi_scale", "train", "shared.training", bool, "Multi-scale jittering", visibility="advanced"),
    _f("freeze", "train", "shared.training", (int, list), "Freeze the first n layers, or explicit layer indices", visibility="advanced"),

    # train -- models.training (the per-model recipe)
    _f("epochs", "train", "models.training", int, "Training epochs", visibility="form", views=("train",)),
    _f("batch_size", "train", "models.training", int, "Training batch size ('auto' on the training CLI probes the largest batch that fits 60% of the GPU memory)", visibility="form", legacy=("batch",), views=("train", "evaluate", "infer", "export")),
    _f("validation_batch_size", "train", "models.training", int, "Validation batch size", visibility="advanced", legacy=("val_batch",)),
    _f("image_size", "train", "models.training", int, "Model input size (a model property; follows the model recipe)", visibility="form", legacy=("imgsz",), views=("train", "evaluate", "infer", "export")),
    _f("optimizer", "train", "models.training", str, "Optimizer (auto, or a concrete name)", visibility="form", views=("train",)),
    _f("learning_rate", "train", "models.training", float, "Initial learning rate", visibility="form", legacy=("lr0",), views=("train",)),
    _f("final_lr_factor", "train", "models.training", float, "Final learning-rate factor", visibility="form", legacy=("lrf",), views=("train",)),
    _f("momentum", "train", "models.training", float, "SGD momentum / Adam beta1", visibility="form", views=("train",)),
    _f("weight_decay", "train", "models.training", float, "Weight decay", visibility="form", views=("train",)),
    _f("warmup_epochs", "train", "models.training", float, "Warmup epochs", visibility="form", views=("train",)),
    _f("warmup_iters", "train", "models.training", int, "Warmup iterations (alternative to warmup_epochs)", visibility="advanced"),
    _f("warmup_start_factor", "train", "models.training", float, "Warmup start learning-rate factor", visibility="advanced"),
    _f("warmup_momentum", "train", "models.training", float, "Warmup start momentum", visibility="advanced", views=("train",)),
    _f("warmup_bias_lr", "train", "models.training", float, "Warmup learning rate for the bias group", visibility="advanced", views=("train",)),
    _f("nominal_batch_size", "train", "models.training", int, "Nominal batch size (used to scale the learning rate)", visibility="advanced", legacy=("nbs",), views=("train",)),
    _f("cosine_schedule", "train", "models.training", bool, "Cosine learning-rate schedule", visibility="form", legacy=("cos_lr",), views=("train",)),
    _f("amp", "train", "models.training", bool, "Mixed-precision training", visibility="form", views=("train",)),
    _f("ema", "train", "models.training", bool, "Exponential moving average", visibility="form"),
    _f("close_mosaic", "train", "models.training", int, "Disable mosaic for the last N epochs", visibility="form", views=("train",)),
    _f("layer_decay_rate", "train", "models.training", float, "ViT layer decay rate", visibility="advanced"),
    _f("seed", "train", "models.training", int, "Recipe-level random seed (beats shared.runtime.seed)", visibility="advanced", views=("train",)),

    # train -- models.loss
    _f("box_loss_weight", "train", "models.loss", float, "Box regression loss weight", visibility="form", legacy=("box",), views=("train",)),
    _f("class_loss_weight", "train", "models.loss", float, "Classification loss weight", visibility="form", legacy=("cls",), views=("train",)),
    _f("distribution_loss_weight", "train", "models.loss", float, "DFL distribution loss weight", visibility="form", legacy=("dfl",), views=("train",)),
    _f("keypoint_loss_weight", "train", "models.loss", float, "Keypoint loss weight", visibility="form", legacy=("pose",), views=("train",)),
    _f("visibility_loss_weight", "train", "models.loss", float, "Visibility loss weight", visibility="form", legacy=("kobj",), views=("train",)),
    _f("label_sigma", "train", "models.loss", float, "SimCC label Gaussian sigma", visibility="advanced"),
    _f("kl_beta", "train", "models.loss", float, "SimCC KL loss beta", visibility="advanced"),

    # train -- models.augmentation (the 13 AnimalRTPose entries)
    _f("hsv_h", "train", "models.augmentation", float, "Hue augmentation gain", visibility="advanced"),
    _f("hsv_s", "train", "models.augmentation", float, "Saturation augmentation gain", visibility="advanced"),
    _f("hsv_v", "train", "models.augmentation", float, "Value augmentation gain", visibility="advanced"),
    _f("degrees", "train", "models.augmentation", float, "Rotation augmentation (+/- degrees)", visibility="advanced"),
    _f("translate", "train", "models.augmentation", float, "Translation augmentation fraction", visibility="advanced"),
    _f("scale", "train", "models.augmentation", float, "Scale augmentation gain", visibility="advanced", legacy=("augment_scale",)),
    _f("shear", "train", "models.augmentation", float, "Shear augmentation (+/- degrees)", visibility="advanced"),
    _f("perspective", "train", "models.augmentation", float, "Perspective augmentation", visibility="advanced"),
    _f("flipud", "train", "models.augmentation", float, "Vertical flip probability", visibility="advanced"),
    _f("fliplr", "train", "models.augmentation", float, "Horizontal flip probability", visibility="advanced"),
    _f("bgr", "train", "models.augmentation", float, "BGR channel-swap probability", visibility="advanced"),
    _f("mosaic", "train", "models.augmentation", float, "Mosaic probability", visibility="advanced"),
    _f("mixup", "train", "models.augmentation", float, "MixUp probability", visibility="advanced"),
    # train -- models.augmentation (the 9 AnimalViTPose entries)
    _f("horizontal_flip_probability", "train", "models.augmentation", float, "Horizontal flip probability", visibility="advanced"),
    _f("half_body_probability", "train", "models.augmentation", float, "Half-body crop probability", visibility="advanced"),
    _f("half_body_upper_probability", "train", "models.augmentation", float, "Probability that a half-body crop keeps the upper body", visibility="advanced"),
    _f("half_body_scale_factor", "train", "models.augmentation", float, "Half-body crop scale factor", visibility="advanced"),
    _f("bbox_shift_probability", "train", "models.augmentation", float, "Box shift probability", visibility="advanced"),
    _f("bbox_shift_factor", "train", "models.augmentation", float, "Box shift factor", visibility="advanced"),
    _f("bbox_scale_factor", "train", "models.augmentation", float, "Box scale factor", visibility="advanced"),
    _f("bbox_rotation_probability", "train", "models.augmentation", float, "Box rotation probability", visibility="advanced"),
    _f("bbox_rotation_factor", "train", "models.augmentation", float, "Box rotation factor (degrees)", visibility="advanced"),

    # train -- models.validation
    _f("confidence", "train", "models.validation", float, "Validation confidence threshold", visibility="advanced", legacy=("conf",), views=("evaluate",)),
    _f("iou", "train", "models.validation", float, "Validation NMS IoU", visibility="advanced", views=("evaluate",)),
    _f("max_det", "train", "models.validation", int, "Validation max detections per image", visibility="advanced", views=("evaluate",)),
    _f("agnostic_nms", "train", "models.validation", bool, "Validation class-agnostic NMS", visibility="advanced", views=("evaluate",)),
    _f("oks_nms_threshold", "train", "models.validation", float, "AnimalViTPose validation OKS-NMS threshold", visibility="advanced"),
    _f("keypoint_score_threshold", "train", "models.validation", float, "AnimalViTPose validation keypoint filter threshold", visibility="advanced"),
    _f("flip_tta", "train", "models.validation", bool, "AnimalViTPose validation flip TTA", visibility="advanced"),

    # train -- metrics
    _f("oks_area_mode", "train", "metrics", str, "OKS area basis: annotation / bbox", visibility="advanced"),
    _f("pck_threshold", "train", "metrics", float, "PCK threshold (as a fraction of the normalized distance)", visibility="advanced"),
    _f("auc_norm_factor", "train", "metrics", float, "AUC normalization factor", visibility="advanced"),
    _f("auc_thresholds", "train", "metrics", int, "AUC threshold sample count", visibility="advanced"),
    _f("coco_max_detections", "train", "metrics", int, "COCO evaluation max detections per image", visibility="advanced"),

    # ------------------------------------------------------------------
    # common
    # ------------------------------------------------------------------
    _f("device", "common", "", str, "Device (auto, or a concrete device name)", visibility="form", views=("train", "evaluate", "infer", "export")),
    _f("project", "common", "", str, "Run output root", visibility="form", views=("train", "evaluate", "infer")),
    _f("name", "common", "", str, "Run name (output goes under project/name)", visibility="form", views=("train", "evaluate", "infer")),
    _f("exist_ok", "common", "", bool, "Allow writing into an existing output directory", visibility="form"),
    _f("selected_weights", "common", "", str, "Explicitly selected weights or artifact path for val/predict/export (the GUI's model picker)", visibility="hidden"),

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    _f("source", "predict", "", str, "Default input source (image / video / directory / glob)", visibility="form", views=("infer",)),
    _f("confidence", "predict", "", float, "Prediction confidence threshold", apply="live", visibility="form", legacy=("prediction_confidence", "conf"), views=("infer",)),
    _f("iou", "predict", "", float, "Prediction NMS IoU", apply="live", visibility="form", legacy=("prediction_iou",), views=("infer",)),
    _f("max_det", "predict", "", int, "Max detections per image", apply="live", visibility="form", views=("infer",)),
    _f("agnostic_nms", "predict", "", bool, "Class-agnostic NMS", apply="live", visibility="form", views=("infer",)),
    _f("keypoint_confidence", "predict", "", float, "Keypoint draw/filter threshold", apply="live", visibility="form", legacy=("keypoint_score_threshold",)),
    _f("dnn", "predict", "", bool, "Infer through the OpenCV DNN backend", visibility="advanced"),
    _f("show_labels", "predict", "", bool, "Draw class labels", apply="live", visibility="form"),
    _f("show_conf", "predict", "", bool, "Draw confidence", apply="live", visibility="form", views=("infer",)),
    _f("show_boxes", "predict", "", bool, "Draw detection boxes", apply="live", visibility="form", views=("infer",)),
    _f("show_keypoints", "predict", "", bool, "Draw keypoints", apply="live", visibility="form"),
    _f("show_skeletons", "predict", "", bool, "Draw skeleton lines", apply="live", visibility="form", legacy=("kpt_line",), views=("infer",)),
    _f("line_width", "predict", "", int, "Box/skeleton line width", apply="live", visibility="form", views=("infer",)),
    _f("point_radius", "predict", "", int, "Keypoint radius", apply="live", visibility="form", legacy=("kpt_radius",), views=("infer",)),
    _f("enabled", "predict", "tracking", bool, "Cross-frame tracking switch", visibility="form", views=("infer",)),
    _f("algorithm", "predict", "tracking", str, "Tracking algorithm (six options, e.g. bytetrack)", visibility="form", views=("infer",)),
    _f("method", "predict", "pose_filter", str, "Temporal smoothing method: none / median / savgol", apply="run_end", visibility="form", views=("infer",)),
    _f("window_length", "predict", "pose_filter", int, "Smoothing window length in frames", apply="run_end", visibility="form", views=("infer",)),
    _f("polyorder", "predict", "pose_filter", int, "Savitzky-Golay polynomial order", apply="run_end", visibility="form", views=("infer",)),
    _f("max_gap", "predict", "pose_filter", int, "Maximum missing-frame gap to interpolate", apply="run_end", visibility="form", views=("infer",)),
    _f("confidence_threshold", "predict", "pose_filter", float, "Filter confidence threshold; null = inherit keypoint_confidence", apply="run_end", visibility="form"),

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    _f("format", "export", "", str, "Export format (onnx / engine / rknn / ...)", visibility="form", views=("export",)),
    _f("half", "export", "", bool, "FP16 export", visibility="form", views=("export",)),
    _f("dynamic", "export", "", bool, "Dynamic axes / dynamic batch", visibility="advanced", views=("export",)),
    _f("simplify", "export", "", bool, "ONNX graph simplification (onnxslim)", visibility="advanced", views=("export",)),
    _f("opset", "export", "", int, "ONNX opset version", visibility="advanced", views=("export",)),
    _f("workspace", "export", "", float, "TensorRT workspace (GiB)", visibility="advanced", views=("export",)),
    _f("rknn_target", "export", "", str, "Rockchip NPU target (e.g. rk3588)", visibility="advanced"),
    _f("optimize", "export", "", bool, "TorchScript mobile optimization", visibility="advanced", views=("export",)),
    _f("int8", "export", "", bool, "INT8 quantization", visibility="advanced", views=("export",)),
    _f("keras", "export", "", bool, "Keras export (errors when unsupported)", visibility="hidden", views=("export",)),
    _f("nms", "export", "", bool, "Embed NMS into the graph (errors when unsupported)", visibility="hidden", views=("export",)),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fields(scope: Optional[str] = None, group: Optional[str] = None) -> Tuple[Field, ...]:
    """All fields, optionally filtered by scope and/or group."""
    return tuple(
        field
        for field in FIELDS
        if (scope is None or field.scope == scope)
        and (group is None or field.group == group)
    )


def field_for(key: str, scope: str, group: str = "") -> Optional[Field]:
    """The field registered under ``(key, scope, group)``, or None."""
    for field in FIELDS:
        if field.key == key and field.scope == scope and field.group == group:
            return field
    return None


def to_cli_name(key: str) -> str:
    """The canonical CLI flag for a key: ``batch_size`` -> ``--batch-size``."""
    return "--" + key.replace("_", "-")


def display_path(field: Field) -> str:
    """Human-readable location used in docs, errors and the GUI."""
    if field.group:
        return f"{field.scope}.{field.group}.{field.key}"
    return f"{field.scope}.{field.key}"


def find_by_legacy(name: str) -> Tuple[Field, ...]:
    """Fields whose legacy spellings include ``name`` (for error hints)."""
    return tuple(field for field in FIELDS if name in field.legacy)
