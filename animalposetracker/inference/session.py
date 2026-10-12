"""Assemble everything needed to run one pose artifact, for every embedder.

The CLI and the GUI page decode the same way: read the artifact metadata, decide which pose
output schema the artifact emits, then hand the decoder an explicit contract. Keeping that
assembly here means the two callers cannot drift apart --
:mod:`animalposetracker.inference.schema_resolution` already promises they behave identically,
and this module is the part of that promise the callers share.

Nothing here trains, and nothing here imports torch: the inferencer is deployment-side code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from animalposetracker.inference.inferencer import InferenceEngine
from animalposetracker.inference.stream_inference import StreamInputConfig


ENGINE_NAMES = {
    "onnx": "ONNX",
    "opencv": "OpenCV",
    "openvino": "OpenVINO",
    "tensorrt": "TensorRT",
    "coreml": "CoreML",
    "cann": "CANN",
}

FORMAT_TO_ENGINE = {
    "onnx": "onnx",
    "engine": "tensorrt",
    "coreml": "coreml",
    "openvino": "openvino",
}


@dataclass(frozen=True)
class InferenceConfigSummary:
    """The class/keypoint picture a config file describes.

    These counts are not decoration: the decoder uses them to cut the raw output tensor into
    "4 box values | one score per class | three values per keypoint", so a config that disagrees
    with the artifact shifts the keypoint block and silently decodes nonsense.
    """

    class_names: Tuple[str, ...] = ()
    keypoints_name: Tuple[str, ...] = ()
    skeleton: Tuple[Tuple[int, int], ...] = ()
    kpt_shape: Optional[Tuple[int, int]] = None
    schema: Optional[str] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_keypoints(self) -> int:
        return self.kpt_shape[0] if self.kpt_shape else len(self.keypoints_name)


@dataclass(frozen=True)
class InferenceSession:
    """A configured inference run, as data.

    Callers that drive their own frame loop (the GUI's staged threads) need the parts rather
    than a pipeline; a caller that wants the whole loop can build
    :class:`~animalposetracker.inference.stream_inference.StreamInferencePipeline` from them.
    """

    engine: InferenceEngine
    output_config: Any
    input_config: StreamInputConfig
    class_names: Tuple[str, ...]
    skeleton: Tuple[Tuple[int, int], ...]
    kpt_shape: Tuple[int, int]
    metadata: Mapping[str, Any]
    head: str
    schema: str
    engine_key: str
    artifact_format: str
    input_size: Tuple[int, int]


def _as_name_list(raw_names) -> List[str]:
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names)]
    return [str(item) for item in (raw_names or [])]


def _parse_inference_config(mapping: Mapping[str, Any]) -> InferenceConfigSummary:
    """Interpret one config mapping, accepting either the project or the dataset spelling.

    A dataset yaml carries ``names``/``kpt_shape``; a project yaml carries
    ``classes_name``/``keypoints_name``/``visible`` and no shape at all, so its keypoint count
    has to be derived. Both are accepted because both are what a user reasonably points at.
    """
    class_names = _as_name_list(mapping.get("names") or mapping.get("classes_name"))
    keypoints_name = tuple(str(name) for name in (mapping.get("keypoints_name") or []))
    skeleton = tuple(
        tuple(int(index) for index in pair) for pair in (mapping.get("skeleton") or [])
    )

    kpt_shape = None
    if mapping.get("kpt_shape"):
        kpt_shape = tuple(int(value) for value in mapping["kpt_shape"])
    else:
        keypoint_count = len(keypoints_name) or int(mapping.get("keypoints") or 0)
        if keypoint_count:
            kpt_shape = (keypoint_count, 3 if mapping.get("visible") else 2)

    raw_schema = mapping.get("schema") or mapping.get("output_schema")
    schema = str(raw_schema).strip().lower() if raw_schema else None
    return InferenceConfigSummary(
        class_names=tuple(class_names),
        keypoints_name=keypoints_name,
        skeleton=skeleton,
        kpt_shape=kpt_shape,
        schema=schema,
    )


def _read_config_mapping(config_arg) -> Optional[Mapping[str, Any]]:
    if config_arg is None:
        return None
    import yaml

    config_path = Path(config_arg).expanduser().resolve()
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def load_inference_config(config_arg):
    """Read the infer-relevant keys from a dataset or project yaml, without insisting on them.

    Returns ``(names, skeleton, kpt_shape, schema)``; ``schema`` is ``None`` when the file does
    not declare one, which lets the caller fall through to the artifact metadata and then to
    shape inference. The CLI relies on that leniency: it can be handed a config that only names
    the classes, with the shape coming from the artifact's sidecar.
    """
    mapping = _read_config_mapping(config_arg)
    if mapping is None:
        return [], [], None, None
    summary = _parse_inference_config(mapping)
    return (
        list(summary.class_names),
        [list(pair) for pair in summary.skeleton],
        summary.kpt_shape,
        summary.schema,
    )


def read_inference_config(config_arg) -> InferenceConfigSummary:
    """Read a config file a human picked, and insist on what the decoder needs.

    The messages are Chinese and name the offending key: they surface directly in the GUI, and
    the failure they describe is impossible to guess from the numpy error it would otherwise
    turn into several layers down.
    """
    if config_arg is None:
        raise ValueError("请先选择配置文件（project.yaml 或 dataset.yaml）。")

    mapping = _read_config_mapping(config_arg)
    if not mapping:
        raise ValueError(f"配置文件是空的或不是 YAML 映射：{config_arg}")

    summary = _parse_inference_config(mapping)
    if not summary.class_names:
        raise ValueError(
            "配置文件缺少类别名：需要 classes_name（项目风格）或 names（数据集风格）。"
        )
    if not summary.num_keypoints:
        raise ValueError(
            "配置文件缺少关键点数量：需要 keypoints_name（项目风格）或 kpt_shape（数据集风格）。"
        )
    if summary.kpt_shape is None:
        raise ValueError("配置文件的关键点数量无法解析：请检查 kpt_shape 或 keypoints_name。")
    return summary


def resolve_num_classes(engine, kpt_shape, default: int = 1, schema: str = None) -> int:
    """Infer the class count from the runtime output tensor width.

    Valid for :data:`~animalposetracker.postprocessing.output_decoders.YOLO_POSE_RAW`
    only: that layout emits ``4 + nc + K * dims`` channels per anchor, so ``nc``
    falls out of the runtime shape without extra metadata.

    Post-NMS and end-to-end layouts have a fixed ``6 + K * dims`` channel count and
    put the *detection count* on the axis read here, so the arithmetic would return a
    meaningless number -- 300 detections over 17x3 keypoints would report 245
    classes. Callers must pass ``schema`` so those layouts take ``nc`` from the
    configured class names instead; ``schema=None`` keeps the historical behaviour
    for callers that predate the parameter.
    """
    if schema is not None:
        import warnings

        from animalposetracker.postprocessing.output_decoders import YOLO_POSE_RAW

        if schema != YOLO_POSE_RAW:
            warnings.warn(
                f"Slot width does not encode the class count for schema {schema!r}; "
                "keeping the supplied default instead of deriving one.",
                stacklevel=2,
            )
            return default
    per_anchor = int(kpt_shape[0]) * int(kpt_shape[1])
    shapes = getattr(engine, "_runtime_output_shapes", {}) or {}
    for shape in shapes.values():
        dims = [int(d) for d in shape]
        if len(dims) >= 3 and dims[1] > 4 + per_anchor:
            return dims[1] - 4 - per_anchor
    return default


def build_inference_session(
    *,
    weights_path,
    config_path=None,
    engine: Optional[str] = None,
    device: str = "CPU",
    bits: str = "FP32",
    conf: float = 0.25,
    iou: float = 0.45,
    agnostic_nms: bool = True,
    schema: Optional[str] = None,
    imgsz: int = 640,
) -> InferenceSession:
    """Load an artifact and work out exactly how to decode what it emits.

    Args:
        weights_path: Exported artifact (``.onnx``/``.engine``/``.mlpackage``/...).
        config_path: Dataset or project yaml, when a class/keypoint source is needed.
        engine: Backend name; defaults to the one implied by the artifact format.
        schema: Explicit pose output schema; ``None``/``"auto"`` defers to the config, then the
            sidecar metadata, then the runtime shapes (see ``schema_resolution``).

    The engine is initialised on the way through, because the schema decision consults the
    runtime output shapes the backend reports.
    """
    from animalposetracker.artifacts import detect_artifact_format, read_artifact_metadata
    from animalposetracker.inference.schema_resolution import resolve_output_schema
    from animalposetracker.postprocessing.output_decoders import (
        YOLO_POSE_RAW,
        PoseOutputConfig,
    )

    weights_path = Path(weights_path)
    metadata = read_artifact_metadata(weights_path)
    artifact_format = detect_artifact_format(weights_path, metadata)
    engine_key = engine or FORMAT_TO_ENGINE.get(artifact_format, "onnx")

    names, skeleton, kpt_shape, config_schema = load_inference_config(config_path)
    if kpt_shape is None and metadata.get("kpt_shape"):
        kpt_shape = tuple(map(int, metadata["kpt_shape"]))
    if kpt_shape is None:
        raise ValueError(
            "kpt_shape is unknown: pass a config yaml with kpt_shape (or a project yaml with "
            "keypoints_name), or use an artifact exported with sidecar metadata"
        )
    input_size = tuple(map(int, metadata.get("input_size") or (imgsz, imgsz)))

    engine_instance = InferenceEngine(
        config=config_path,
        weights_path=str(weights_path),
        input_width=input_size[0],
        input_height=input_size[1],
        engine=ENGINE_NAMES[engine_key],
        device=device,
        model_bits=bits,
        conf=conf,
        iou=iou,
        output_names=tuple(metadata.get("outputs") or ()),
    )
    engine_instance.model_init()

    head = str(metadata.get("head") or "")
    if not head:
        head = "SimCCHead" if "simcc_x" in tuple(metadata.get("outputs") or ()) else "YOLOPoseHead"
    if head == "SimCCHead":
        raise ValueError(
            f"{weights_path} holds a SimCCHead (AnimalViTPose) model, and the inferencer cannot "
            "run one: top-down decoding needs a separate detector, which it has no way to "
            "supply. Run it through `animalposetracker predict` instead."
        )

    # A YOLOPoseHead artifact can emit any of the three ultralytics layouts. The explicit
    # argument wins over the config yaml, which wins over the sidecar, which wins over the
    # runtime shape; see schema_resolution for the rules.
    resolved_schema, inferred_num_classes = resolve_output_schema(
        explicit_schema=schema if schema not in (None, "", "auto") else config_schema,
        metadata=metadata,
        output_shapes=getattr(engine_instance, "_runtime_output_shapes", {}) or {},
        num_keypoints=kpt_shape[0],
        keypoint_dims=kpt_shape[1],
    )
    if inferred_num_classes is not None:
        num_classes = inferred_num_classes
    elif resolved_schema == YOLO_POSE_RAW:
        num_classes = resolve_num_classes(engine_instance, kpt_shape, schema=resolved_schema)
    else:
        # Post-NMS and end-to-end rows carry the class id directly, so the only thing
        # num_classes feeds is the class-name lookup in the rendered records. The configured
        # names are the right source.
        num_classes = max(1, len(names))

    output_config = PoseOutputConfig(
        schema=resolved_schema,
        num_classes=num_classes,
        num_keypoints=kpt_shape[0],
        keypoint_dims=kpt_shape[1],
        # The decoder looks tensors up by name, and the engine already asked the backend which
        # names exist. Handing them over keeps the two in step: a foreign artifact names its
        # output "output0" (ultralytics) rather than the "predictions" default, so without this
        # an artifact that carries no sidecar could not be decoded at all. An empty tuple here
        # falls back to the schema default, exactly as before.
        output_names=tuple(engine_instance.output_names or ()),
        confidence_threshold=conf,
        iou_threshold=iou,
        class_agnostic_nms=agnostic_nms,
    )
    input_config = StreamInputConfig(
        input_size=input_size,
        preprocess_mode="engine",
        color_order=str(metadata.get("color_order", "RGB")),
    )
    return InferenceSession(
        engine=engine_instance,
        output_config=output_config,
        input_config=input_config,
        class_names=tuple(names),
        skeleton=tuple(tuple(int(index) for index in pair) for pair in skeleton),
        kpt_shape=tuple(int(value) for value in kpt_shape),
        metadata=metadata,
        head=head,
        schema=resolved_schema,
        engine_key=engine_key,
        artifact_format=artifact_format,
        input_size=input_size,
    )


def instance_to_record(instance, class_names) -> Dict[str, Any]:
    """Turn one decoded instance into the record shape tracking and rendering expect."""
    keypoints = instance.keypoints_xy
    scores = instance.keypoint_scores
    if scores is None:
        scores = instance.keypoints_visible
    if scores is not None:
        import numpy as np

        scores_col = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
        if scores_col.shape[0] == keypoints.shape[0]:
            keypoints = np.concatenate([keypoints, scores_col], axis=1)
    class_id = int(instance.class_id) if instance.class_id is not None else 0
    return {
        "bbox_xyxy": None if instance.bbox_xyxy is None else [
            float(v) for v in instance.bbox_xyxy
        ],
        "score": float(instance.score) if instance.score is not None else 1.0,
        "class_id": class_id,
        "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
        "keypoints": keypoints.tolist(),
    }


__all__ = [
    "ENGINE_NAMES",
    "FORMAT_TO_ENGINE",
    "InferenceConfigSummary",
    "InferenceSession",
    "build_inference_session",
    "instance_to_record",
    "load_inference_config",
    "read_inference_config",
    "resolve_num_classes",
]
