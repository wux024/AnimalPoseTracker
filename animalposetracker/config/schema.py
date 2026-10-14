"""The configuration schema: one registry entry per configuration key.

This module is the single place where a configuration key is *defined* -- its
canonical name, which scope and group it lives in, its type, when it takes
effect and whether the GUI shows it. Default *values* live in the defaults
files (``cfg/training.yaml``, ``cfg/runtime_defaults.yaml``); this registry
describes the keys those files and the project's ``runtime.yaml`` may carry.

The registry is data, not code: the resolver, the CLI builders, the GUI and
the generated documentation all read it, so none of them can drift from the
others (a key renamed here is renamed everywhere).

Terminology (mirrors 配置概念字典-W0-2026-10-14.md):

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
- ``visibility``: where the GUI shows the key -- ``form`` (常规),
  ``advanced`` (高级/折叠), ``hidden`` (不进 GUI).
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
) -> Field:
    return Field(key, scope, group, type, note, apply, visibility, tuple(legacy))


# ---------------------------------------------------------------------------
# train -- shared.runtime (运行类)
# ---------------------------------------------------------------------------

FIELDS: Tuple[Field, ...] = (
    _f("model", "train", "shared.runtime", str, "模型图路径（由项目结构推导，不落盘）", visibility="hidden"),
    _f("data", "train", "shared.runtime", str, "数据集路径（由项目结构推导，不落盘）", visibility="hidden"),
    _f("num_workers", "train", "shared.runtime", int, "数据加载线程数", visibility="form", legacy=("workers",)),
    _f("time_hours", "train", "shared.runtime", float, "训练时间上限（小时），给出后覆盖 epochs", visibility="advanced", legacy=("time",)),
    _f("cache", "train", "shared.runtime", (bool, str), "数据缓存：false / true(ram) / ram / disk", visibility="form"),
    _f("fraction", "train", "shared.runtime", float, "训练集使用比例", visibility="form"),
    _f("profile", "train", "shared.runtime", bool, "逐阶段耗时剖析", visibility="advanced"),
    _f("plots", "train", "shared.runtime", bool, "保存训练曲线图", visibility="form"),
    _f("validation_enabled", "train", "shared.runtime", bool, "训练时是否跑验证", visibility="form", legacy=("val",)),
    _f("seed", "train", "shared.runtime", int, "随机种子", visibility="form"),
    _f("deterministic", "train", "shared.runtime", bool, "确定性算法", visibility="advanced"),
    _f("save", "train", "shared.runtime", bool, "保存 checkpoint", visibility="form"),
    _f("save_period", "train", "shared.runtime", int, "每 N 轮另存 epoch checkpoint；0 = 关闭", visibility="advanced"),
    _f("pretrained", "train", "shared.runtime", bool, "使用预训练权重", visibility="form"),
    _f("pretrained_weights", "train", "shared.runtime", str, "显式预训练权重路径", visibility="advanced"),
    _f("verbose", "train", "shared.runtime", bool, "详细日志", visibility="advanced"),
    _f("resume", "train", "shared.runtime", (bool, str), "从 last checkpoint 继续训练", visibility="form"),
    _f("resume_from", "train", "shared.runtime", str, "从指定完整 checkpoint 恢复", visibility="hidden"),
    _f("channels_last", "train", "shared.runtime", bool, "channels_last 显式开关；null = 自动", visibility="advanced"),
    _f("gradient_accumulation_steps", "train", "shared.runtime", int, "梯度累积步数", visibility="advanced"),
    _f("gradient_clip_norm", "train", "shared.runtime", float, "梯度裁剪范数；null = 关闭", visibility="advanced"),
    _f("lr_milestones", "train", "shared.runtime", list, "MultiStep 学习率里程碑", visibility="advanced"),
    _f("lr_gamma", "train", "shared.runtime", float, "里程碑处的学习率衰减", visibility="advanced"),
    _f("validation_interval", "train", "shared.runtime", int, "每 N 轮验证一次", visibility="advanced"),
    _f("log_interval", "train", "shared.runtime", int, "每 N 个 batch 打日志", visibility="advanced"),
    _f("best_metric", "train", "shared.runtime", str, "最优模型选择指标（coco/AP 或 ultralytics/AP）", visibility="advanced"),
    _f("best_metric_mode", "train", "shared.runtime", str, "选择方向：max / min", visibility="advanced"),
    _f("early_stopping_patience", "train", "shared.runtime", int, "早停耐心轮数", visibility="form", legacy=("patience",)),

    # train -- shared.training
    _f("single_cls", "train", "shared.training", bool, "多类数据按单类训练", visibility="advanced"),
    _f("multi_scale", "train", "shared.training", bool, "多尺度抖动", visibility="advanced"),
    _f("freeze", "train", "shared.training", (int, list), "冻结前 n 层或指定层索引", visibility="advanced"),

    # train -- models.training（逐模型配方）
    _f("epochs", "train", "models.training", int, "训练轮数", visibility="form"),
    _f("batch_size", "train", "models.training", int, "训练批大小", visibility="form", legacy=("batch",)),
    _f("validation_batch_size", "train", "models.training", int, "验证批大小", visibility="advanced", legacy=("val_batch",)),
    _f("image_size", "train", "models.training", int, "模型输入尺寸（模型属性：跟随模型配方）", visibility="form", legacy=("imgsz",)),
    _f("optimizer", "train", "models.training", str, "优化器（auto 或具体名字）", visibility="form"),
    _f("learning_rate", "train", "models.training", float, "初始学习率", visibility="form", legacy=("lr0",)),
    _f("final_lr_factor", "train", "models.training", float, "末端学习率因子", visibility="form", legacy=("lrf",)),
    _f("momentum", "train", "models.training", float, "SGD 动量 / Adam beta1", visibility="form"),
    _f("weight_decay", "train", "models.training", float, "权重衰减", visibility="form"),
    _f("warmup_epochs", "train", "models.training", float, "warmup 轮数", visibility="form"),
    _f("warmup_iters", "train", "models.training", int, "warmup 步数（与 warmup_epochs 二选一）", visibility="advanced"),
    _f("warmup_start_factor", "train", "models.training", float, "warmup 起始学习率因子", visibility="advanced"),
    _f("warmup_momentum", "train", "models.training", float, "warmup 起始动量", visibility="advanced"),
    _f("warmup_bias_lr", "train", "models.training", float, "bias 组 warmup 学习率", visibility="advanced"),
    _f("nominal_batch_size", "train", "models.training", int, "名义批大小（缩放学习率用）", visibility="advanced", legacy=("nbs",)),
    _f("cosine_schedule", "train", "models.training", bool, "余弦学习率调度", visibility="form", legacy=("cos_lr",)),
    _f("amp", "train", "models.training", bool, "混合精度训练", visibility="form"),
    _f("ema", "train", "models.training", bool, "指数滑动平均", visibility="form"),
    _f("close_mosaic", "train", "models.training", int, "末尾 N 轮关闭 mosaic", visibility="form"),
    _f("layer_decay_rate", "train", "models.training", float, "ViT 层衰减率", visibility="advanced"),
    _f("seed", "train", "models.training", int, "模型配方级随机种子（优先于 shared.runtime.seed）", visibility="advanced"),

    # train -- models.loss
    _f("box_loss_weight", "train", "models.loss", float, "框回归损失权重", visibility="form", legacy=("box",)),
    _f("class_loss_weight", "train", "models.loss", float, "分类损失权重", visibility="form", legacy=("cls",)),
    _f("distribution_loss_weight", "train", "models.loss", float, "DFL 分布损失权重", visibility="form", legacy=("dfl",)),
    _f("keypoint_loss_weight", "train", "models.loss", float, "关键点损失权重", visibility="form", legacy=("pose",)),
    _f("visibility_loss_weight", "train", "models.loss", float, "可见性损失权重", visibility="form", legacy=("kobj",)),
    _f("label_sigma", "train", "models.loss", float, "SimCC 标签高斯 sigma", visibility="advanced"),
    _f("kl_beta", "train", "models.loss", float, "SimCC KL 损失 beta", visibility="advanced"),

    # train -- models.augmentation（RTPose 13 项）
    _f("hsv_h", "train", "models.augmentation", float, "色调增强幅度", visibility="advanced"),
    _f("hsv_s", "train", "models.augmentation", float, "饱和度增强幅度", visibility="advanced"),
    _f("hsv_v", "train", "models.augmentation", float, "明度增强幅度", visibility="advanced"),
    _f("degrees", "train", "models.augmentation", float, "旋转增强（±度）", visibility="advanced"),
    _f("translate", "train", "models.augmentation", float, "平移增强比例", visibility="advanced"),
    _f("scale", "train", "models.augmentation", float, "缩放增强增益", visibility="advanced", legacy=("augment_scale",)),
    _f("shear", "train", "models.augmentation", float, "剪切增强（±度）", visibility="advanced"),
    _f("perspective", "train", "models.augmentation", float, "透视增强", visibility="advanced"),
    _f("flipud", "train", "models.augmentation", float, "上下翻转概率", visibility="advanced"),
    _f("fliplr", "train", "models.augmentation", float, "左右翻转概率", visibility="advanced"),
    _f("bgr", "train", "models.augmentation", float, "BGR 通道交换概率", visibility="advanced"),
    _f("mosaic", "train", "models.augmentation", float, "mosaic 概率", visibility="advanced"),
    _f("mixup", "train", "models.augmentation", float, "mixup 概率", visibility="advanced"),
    # train -- models.augmentation（ViTPose 9 项）
    _f("horizontal_flip_probability", "train", "models.augmentation", float, "水平翻转概率", visibility="advanced"),
    _f("half_body_probability", "train", "models.augmentation", float, "半身裁剪概率", visibility="advanced"),
    _f("half_body_upper_probability", "train", "models.augmentation", float, "半身裁剪取上半身概率", visibility="advanced"),
    _f("half_body_scale_factor", "train", "models.augmentation", float, "半身裁剪缩放因子", visibility="advanced"),
    _f("bbox_shift_probability", "train", "models.augmentation", float, "框平移概率", visibility="advanced"),
    _f("bbox_shift_factor", "train", "models.augmentation", float, "框平移幅度", visibility="advanced"),
    _f("bbox_scale_factor", "train", "models.augmentation", float, "框缩放幅度", visibility="advanced"),
    _f("bbox_rotation_probability", "train", "models.augmentation", float, "框旋转概率", visibility="advanced"),
    _f("bbox_rotation_factor", "train", "models.augmentation", float, "框旋转幅度（度）", visibility="advanced"),

    # train -- models.validation
    _f("confidence", "train", "models.validation", float, "验证置信度阈值", visibility="advanced", legacy=("conf",)),
    _f("iou", "train", "models.validation", float, "验证 NMS IoU", visibility="advanced"),
    _f("max_det", "train", "models.validation", int, "验证每图最大检测数", visibility="advanced"),
    _f("agnostic_nms", "train", "models.validation", bool, "验证类无关 NMS", visibility="advanced"),
    _f("oks_nms_threshold", "train", "models.validation", float, "ViTPose 验证 OKS-NMS 阈值", visibility="advanced"),
    _f("keypoint_score_threshold", "train", "models.validation", float, "ViTPose 验证关键点过滤阈值", visibility="advanced"),
    _f("flip_tta", "train", "models.validation", bool, "ViTPose 验证翻转 TTA", visibility="advanced"),

    # train -- metrics
    _f("oks_area_mode", "train", "metrics", str, "OKS 面积口径：annotation / bbox", visibility="advanced"),
    _f("pck_threshold", "train", "metrics", float, "PCK 判定阈值（占归一化距离比例）", visibility="advanced"),
    _f("auc_norm_factor", "train", "metrics", float, "AUC 归一化因子", visibility="advanced"),
    _f("auc_thresholds", "train", "metrics", int, "AUC 阈值采样数", visibility="advanced"),
    _f("coco_max_detections", "train", "metrics", int, "COCO 评估每图最大检测数", visibility="advanced"),

    # ------------------------------------------------------------------
    # common
    # ------------------------------------------------------------------
    _f("device", "common", "", str, "运行设备（auto 或具体设备名）", visibility="form"),
    _f("project", "common", "", str, "运行输出根目录", visibility="form"),
    _f("name", "common", "", str, "运行名（输出在 project/name 下）", visibility="form"),
    _f("exist_ok", "common", "", bool, "允许写进已存在的输出目录", visibility="form"),
    _f("selected_weights", "common", "", str, "为评估/预测/导出显式选择的权重或产物路径（GUI 模型选择）", visibility="hidden"),

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    _f("source", "predict", "", str, "默认输入源（图片/视频/目录/glob）", visibility="form"),
    _f("confidence", "predict", "", float, "预测置信度阈值", apply="live", visibility="form", legacy=("prediction_confidence",)),
    _f("iou", "predict", "", float, "预测 NMS IoU", apply="live", visibility="form", legacy=("prediction_iou",)),
    _f("max_det", "predict", "", int, "每图最大检测数", apply="live", visibility="form"),
    _f("agnostic_nms", "predict", "", bool, "类无关 NMS", apply="live", visibility="form"),
    _f("keypoint_confidence", "predict", "", float, "关键点绘制/过滤阈值", apply="live", visibility="form", legacy=("keypoint_score_threshold",)),
    _f("dnn", "predict", "", bool, "用 OpenCV DNN 后端推理", visibility="advanced"),
    _f("show_labels", "predict", "", bool, "绘制类标签", apply="live", visibility="form"),
    _f("show_conf", "predict", "", bool, "绘制置信度", apply="live", visibility="form"),
    _f("show_boxes", "predict", "", bool, "绘制检测框", apply="live", visibility="form"),
    _f("show_keypoints", "predict", "", bool, "绘制关键点", apply="live", visibility="form"),
    _f("show_skeletons", "predict", "", bool, "绘制骨架连线", apply="live", visibility="form", legacy=("kpt_line",)),
    _f("line_width", "predict", "", int, "框/骨架线宽", apply="live", visibility="form"),
    _f("point_radius", "predict", "", int, "关键点半径", apply="live", visibility="form", legacy=("kpt_radius",)),
    _f("enabled", "predict", "tracking", bool, "跨帧跟踪开关", visibility="form"),
    _f("algorithm", "predict", "tracking", str, "跟踪算法（bytetrack 等 6 种）", visibility="form"),
    _f("method", "predict", "pose_filter", str, "时序平滑方法：none / median / savgol", apply="run_end", visibility="form"),
    _f("window_length", "predict", "pose_filter", int, "平滑窗口帧数", apply="run_end", visibility="form"),
    _f("polyorder", "predict", "pose_filter", int, "Savitzky-Golay 多项式阶数", apply="run_end", visibility="form"),
    _f("max_gap", "predict", "pose_filter", int, "最大插值缺失帧数", apply="run_end", visibility="form"),
    _f("confidence_threshold", "predict", "pose_filter", float, "滤波可信阈值；null = 继承 keypoint_confidence", apply="run_end", visibility="form"),

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    _f("format", "export", "", str, "导出格式（onnx / engine / rknn ...）", visibility="form"),
    _f("half", "export", "", bool, "FP16 导出", visibility="form"),
    _f("dynamic", "export", "", bool, "动态轴/动态 batch", visibility="advanced"),
    _f("simplify", "export", "", bool, "ONNX 图简化（onnxslim）", visibility="advanced"),
    _f("opset", "export", "", int, "ONNX opset 版本", visibility="advanced"),
    _f("workspace", "export", "", float, "TensorRT workspace（GiB）", visibility="advanced"),
    _f("rknn_target", "export", "", str, "Rockchip NPU 目标（如 rk3588）", visibility="advanced"),
    _f("optimize", "export", "", bool, "TorchScript 移动端优化", visibility="advanced"),
    _f("int8", "export", "", bool, "INT8 量化", visibility="advanced"),
    _f("keras", "export", "", bool, "Keras 导出（不支持则报错）", visibility="hidden"),
    _f("nms", "export", "", bool, "图中嵌入 NMS（不支持则报错）", visibility="hidden"),
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
