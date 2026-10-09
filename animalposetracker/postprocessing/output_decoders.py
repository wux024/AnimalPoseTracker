"""Output schemas and frame-space decoders for live pose inference.

This module deliberately knows nothing about model execution backends. Runtime
adapters normalize backend-specific return containers, then these decoders turn
named tensors into a common per-instance result.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


YOLO_POSE_RAW = "ultralytics_yolo_pose_raw"
YOLO_POSE_NMS = "ultralytics_yolo_pose_nms"
YOLO_POSE_END2END = "ultralytics_yolo_pose_end2end"
RTMO_MMDEPLOY = "mmpose_rtmo_mmdeploy"
RTMPOSE_SIMCC = "mmpose_rtmpose_simcc"

SUPPORTED_OUTPUT_SCHEMAS = (
    YOLO_POSE_RAW,
    YOLO_POSE_NMS,
    YOLO_POSE_END2END,
    RTMO_MMDEPLOY,
    RTMPOSE_SIMCC,
)

_DEFAULT_OUTPUT_NAMES = {
    YOLO_POSE_RAW: ("predictions",),
    YOLO_POSE_NMS: ("predictions",),
    YOLO_POSE_END2END: ("predictions",),
    RTMO_MMDEPLOY: ("dets", "pred_kpts"),
    RTMPOSE_SIMCC: ("simcc_x", "simcc_y"),
}


@dataclass(frozen=True)
class PoseInstance:
    """One decoded pose in original stream-frame coordinates."""

    keypoints_xy: np.ndarray
    bbox_xyxy: Optional[np.ndarray] = None
    score: Optional[float] = None
    class_id: Optional[int] = None
    keypoint_scores: Optional[np.ndarray] = None
    keypoints_visible: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PoseOutputConfig:
    """The explicit output contract needed to decode a pose model."""

    schema: str
    num_keypoints: int
    num_classes: int = 1
    keypoint_dims: int = 3
    output_names: Tuple[str, ...] = ()
    output_layout: str = "BCN"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_agnostic_nms: bool = True
    simcc_split_ratio: float = 2.0
    keypoint_channel_semantics: str = "visibility"

    def __post_init__(self):
        object.__setattr__(self, "output_names", tuple(self.output_names or ()))
        if self.schema not in SUPPORTED_OUTPUT_SCHEMAS:
            raise ValueError(
                f"Unsupported pose output schema {self.schema!r}; choose from "
                f"{', '.join(SUPPORTED_OUTPUT_SCHEMAS)}"
            )
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.keypoint_dims not in (2, 3):
            raise ValueError("keypoint_dims must be 2 or 3")
        if self.output_layout not in ("BCN", "BNC"):
            raise ValueError("output_layout must be 'BCN' or 'BNC'")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if self.simcc_split_ratio <= 0.0:
            raise ValueError("simcc_split_ratio must be positive")
        if self.keypoint_channel_semantics not in ("confidence", "visibility"):
            raise ValueError(
                "keypoint_channel_semantics must be 'confidence' or 'visibility'"
            )
        expected_outputs = 2 if self.schema in (RTMO_MMDEPLOY, RTMPOSE_SIMCC) else 1
        if self.output_names and len(self.output_names) != expected_outputs:
            raise ValueError(
                f"Schema {self.schema!r} expects {expected_outputs} output names, "
                f"got {len(self.output_names)}"
            )

    @property
    def resolved_output_names(self) -> Tuple[str, ...]:
        return self.output_names or _DEFAULT_OUTPUT_NAMES[self.schema]


def _output_name(key: Any, index: int) -> str:
    if isinstance(key, str):
        return key
    for attr in ("any_name", "get_any_name", "name"):
        value = getattr(key, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if value:
            return str(value)
    return f"output_{index}"


def normalize_output_tensors(
    outputs: Any,
    output_names: Sequence[str] = (),
) -> Mapping[str, np.ndarray]:
    """Normalize arrays, sequences, and backend output mappings by name."""
    names = tuple(str(name) for name in output_names)
    if isinstance(outputs, Mapping):
        named = {
            _output_name(key, index): np.asarray(value)
            for index, (key, value) in enumerate(outputs.items())
        }
        if names:
            missing = [name for name in names if name not in named]
            if missing:
                # Some backend maps use output-port objects whose display name
                # differs from the exported tensor name. Keep their stable order
                # when the number of requested and returned tensors agrees.
                values = list(outputs.values())
                if len(values) != len(names):
                    raise ValueError(
                        f"Model outputs are missing configured names {missing}; "
                        f"available outputs: {list(named)}"
                    )
                return {name: np.asarray(value) for name, value in zip(names, values)}
            return {name: named[name] for name in names}
        return named

    if isinstance(outputs, (tuple, list)):
        values = [np.asarray(value) for value in outputs]
    else:
        values = [np.asarray(outputs)]

    if names and len(names) != len(values):
        raise ValueError(
            f"Configured {len(names)} output names for {len(values)} output tensors"
        )
    resolved_names = names or tuple(f"output_{index}" for index in range(len(values)))
    return {name: value for name, value in zip(resolved_names, values)}


def _get_output(
    outputs: Mapping[str, np.ndarray], config: PoseOutputConfig, index: int = 0
) -> np.ndarray:
    name = config.resolved_output_names[index]
    if name not in outputs:
        raise ValueError(
            f"Output schema {config.schema!r} requires tensor {name!r}; "
            f"available tensors: {list(outputs)}"
        )
    return np.asarray(outputs[name])


def _single_batch(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 0:
        raise ValueError(f"Output {name!r} must have at least one dimension")
    if array.ndim >= 2:
        if array.shape[0] != 1:
            raise ValueError(
                f"Live stream decoding expects batch size 1 for {name!r}, "
                f"got shape {array.shape}"
            )
        return array[0]
    return array


def _apply_inverse(points: np.ndarray, inverse_affine: Optional[np.ndarray]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    if inverse_affine is None or points.size == 0:
        return points
    source_x = points[..., 0].copy()
    source_y = points[..., 1].copy()
    points[..., 0] = (
        source_x * inverse_affine[0, 0]
        + source_y * inverse_affine[0, 1]
        + inverse_affine[0, 2]
    )
    points[..., 1] = (
        source_x * inverse_affine[1, 0]
        + source_y * inverse_affine[1, 1]
        + inverse_affine[1, 2]
    )
    return points


def _apply_inverse_boxes(
    boxes_xyxy: np.ndarray, inverse_affine: Optional[np.ndarray]
) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if inverse_affine is None or boxes.size == 0:
        return boxes
    corners = np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1,
    )
    mapped = _apply_inverse(corners, inverse_affine)
    boxes[:, :2] = mapped.min(axis=1)
    boxes[:, 2:] = mapped.max(axis=1)
    return boxes


def _keypoint_fields(
    flat_values: np.ndarray, config: PoseOutputConfig, inverse_affine: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    keypoints = np.asarray(flat_values, dtype=np.float32).reshape(
        config.num_keypoints, config.keypoint_dims
    )
    keypoints_xy = _apply_inverse(keypoints[:, :2], inverse_affine)
    third_channel = keypoints[:, 2].copy() if config.keypoint_dims == 3 else None
    if third_channel is None:
        return keypoints_xy, None, None
    if config.keypoint_channel_semantics == "visibility":
        return keypoints_xy, None, third_channel
    return keypoints_xy, third_channel, None


def _nms_indices(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    config: PoseOutputConfig,
) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)
    keep: List[int] = []
    class_groups = [None] if config.class_agnostic_nms else np.unique(class_ids)
    for class_id in class_groups:
        indices = (
            np.arange(len(boxes_xyxy), dtype=np.int64)
            if class_id is None
            else np.flatnonzero(class_ids == class_id)
        )
        if not len(indices):
            continue
        boxes = boxes_xyxy[indices]
        xywh = np.column_stack(
            [
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ]
        )
        selected = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores[indices].astype(float).tolist(),
            float(config.confidence_threshold),
            float(config.iou_threshold),
        )
        if selected is not None:
            keep.extend(indices[np.asarray(selected, dtype=np.int64).reshape(-1)].tolist())
    if not keep:
        return np.empty((0,), dtype=np.int64)
    keep_array = np.asarray(keep, dtype=np.int64)
    return keep_array[np.argsort(scores[keep_array])[::-1]]


def _decode_yolo_raw(
    outputs: Mapping[str, np.ndarray],
    config: PoseOutputConfig,
    inverse_affine: Optional[np.ndarray],
) -> List[PoseInstance]:
    output = _single_batch(_get_output(outputs, config), config.resolved_output_names[0])
    if output.ndim != 2:
        raise ValueError(f"YOLO raw output must be rank 2 after batch removal, got {output.shape}")
    rows = output.T if config.output_layout == "BCN" else output
    expected = 4 + config.num_classes + config.num_keypoints * config.keypoint_dims
    if rows.shape[1] != expected:
        raise ValueError(
            f"YOLO raw output has {rows.shape[1]} channels; output config expects {expected}"
        )

    class_scores = rows[:, 4:4 + config.num_classes]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
    scores = class_scores[np.arange(len(rows)), class_ids].astype(np.float32)
    selected = (scores > 0.0) & (scores >= config.confidence_threshold)
    rows = rows[selected]
    class_ids = class_ids[selected]
    scores = scores[selected]
    boxes = rows[:, :4].astype(np.float32)
    centers = boxes[:, :2]
    half_sizes = boxes[:, 2:4] * 0.5
    boxes_xyxy = np.concatenate([centers - half_sizes, centers + half_sizes], axis=1)
    boxes_xyxy = _apply_inverse_boxes(boxes_xyxy, inverse_affine)
    keypoint_data = rows[:, 4 + config.num_classes:]
    decoded_keypoints = [
        _keypoint_fields(values, config, inverse_affine) for values in keypoint_data
    ]
    keep = _nms_indices(boxes_xyxy, scores, class_ids, config)
    return [
        PoseInstance(
            bbox_xyxy=boxes_xyxy[index],
            score=float(scores[index]),
            class_id=int(class_ids[index]),
            keypoints_xy=decoded_keypoints[index][0],
            keypoint_scores=decoded_keypoints[index][1],
            keypoints_visible=decoded_keypoints[index][2],
        )
        for index in keep
    ]


def _decode_yolo_processed(
    outputs: Mapping[str, np.ndarray],
    config: PoseOutputConfig,
    inverse_affine: Optional[np.ndarray],
) -> List[PoseInstance]:
    output = _single_batch(_get_output(outputs, config), config.resolved_output_names[0])
    if output.ndim != 2:
        raise ValueError(
            f"Processed YOLO output must be rank 2 after batch removal, got {output.shape}"
        )
    # Ultralytics processed rows: xyxy, confidence, class id, flattened keypoints.
    expected = 6 + config.num_keypoints * config.keypoint_dims
    if output.shape[1] != expected:
        raise ValueError(
            f"Processed YOLO output has {output.shape[1]} columns; output config expects {expected}"
        )
    output = output[(output[:, 4] > 0.0) & (output[:, 4] >= config.confidence_threshold)]
    if not len(output):
        return []
    boxes = _apply_inverse_boxes(output[:, :4], inverse_affine)
    instances = []
    for row, bbox in zip(output, boxes):
        keypoints_xy, keypoint_scores, keypoints_visible = _keypoint_fields(
            row[6:], config, inverse_affine
        )
        instances.append(PoseInstance(
            bbox_xyxy=bbox,
            score=float(row[4]),
            class_id=int(row[5]),
            keypoints_xy=keypoints_xy,
            keypoint_scores=keypoint_scores,
            keypoints_visible=keypoints_visible,
        ))
    return instances


def _decode_rtmo_mmdeploy(
    outputs: Mapping[str, np.ndarray],
    config: PoseOutputConfig,
    inverse_affine: Optional[np.ndarray],
) -> List[PoseInstance]:
    dets = _single_batch(_get_output(outputs, config, 0), config.resolved_output_names[0])
    keypoints = _single_batch(_get_output(outputs, config, 1), config.resolved_output_names[1])
    if dets.ndim != 2 or keypoints.ndim != 3:
        raise ValueError(
            "RTMO MMDeploy outputs must have shapes [N, 4+scores] and [N, K, 2/3] "
            f"after batch removal; got {dets.shape} and {keypoints.shape}"
        )
    if len(dets) != len(keypoints):
        raise ValueError("RTMO dets and pred_kpts instance counts do not match")
    if keypoints.shape[1] != config.num_keypoints or keypoints.shape[2] not in (2, 3):
        raise ValueError(
            f"RTMO pred_kpts shape {keypoints.shape} does not match "
            f"K={config.num_keypoints} and keypoint dimension 2/3"
        )

    if dets.shape[1] == 5:
        boxes = dets[:, :4].astype(np.float32)
        scores = dets[:, 4].astype(np.float32)
        class_ids = np.zeros(len(dets), dtype=np.int64)
    elif dets.shape[1] == 4 + config.num_classes:
        class_scores = dets[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
        scores = class_scores[np.arange(len(dets)), class_ids]
        boxes = dets[:, :4].astype(np.float32)
    else:
        raise ValueError(
            f"RTMO dets has {dets.shape[1]} columns; expected xyxy+score "
            f"or xyxy+{config.num_classes} class scores"
        )
    keep = (scores > 0.0) & (scores >= config.confidence_threshold)
    boxes, scores, class_ids, keypoints = boxes[keep], scores[keep], class_ids[keep], keypoints[keep]
    boxes = _apply_inverse_boxes(boxes, inverse_affine)
    keypoints = keypoints.astype(np.float32)
    result = []
    for index, (bbox, points) in enumerate(zip(boxes, keypoints)):
        keypoints_xy = _apply_inverse(points[:, :2], inverse_affine)
        visibility = points[:, 2].copy() if points.shape[1] == 3 else None
        result.append(PoseInstance(
            bbox_xyxy=bbox,
            score=float(scores[index]),
            class_id=int(class_ids[index]),
            keypoints_xy=keypoints_xy,
            keypoint_scores=None,
            keypoints_visible=visibility,
        ))
    return result


def decode_simcc(
    outputs: Mapping[str, np.ndarray],
    config: PoseOutputConfig,
    inverse_affine: Optional[np.ndarray] = None,
    bbox_xyxy: Optional[np.ndarray] = None,
    bbox_score: Optional[float] = None,
    class_id: Optional[int] = None,
) -> PoseInstance:
    """Decode one top-down SimCC crop and map it to the original stream frame."""
    simcc_x = _single_batch(_get_output(outputs, config, 0), config.resolved_output_names[0])
    simcc_y = _single_batch(_get_output(outputs, config, 1), config.resolved_output_names[1])
    if simcc_x.ndim != 2 or simcc_x.shape[0] != config.num_keypoints:
        raise ValueError(f"simcc_x must have shape [K, Wx], got {simcc_x.shape}")
    if simcc_y.ndim != 2 or simcc_y.shape[0] != config.num_keypoints:
        raise ValueError(f"simcc_y must have shape [K, Wy], got {simcc_y.shape}")
    x_indices = np.argmax(simcc_x, axis=-1)
    y_indices = np.argmax(simcc_y, axis=-1)
    scores = np.minimum(
        np.max(simcc_x, axis=-1),
        np.max(simcc_y, axis=-1),
    ).astype(np.float32)
    points = np.stack([x_indices, y_indices], axis=-1).astype(np.float32)
    points /= float(config.simcc_split_ratio)
    points[scores <= 0.0] = -1.0 / float(config.simcc_split_ratio)
    points = _apply_inverse(points, inverse_affine)
    box = None
    if bbox_xyxy is not None:
        box = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4).copy()
    return PoseInstance(
        bbox_xyxy=box,
        score=float(np.mean(scores)) if bbox_score is None else float(bbox_score),
        class_id=class_id,
        keypoints_xy=points,
        keypoint_scores=scores,
    )


def decode_pose_outputs(
    outputs: Any,
    config: PoseOutputConfig,
    inverse_affine: Optional[np.ndarray] = None,
) -> List[PoseInstance]:
    """Decode a named backend output according to an explicit output schema."""
    named_outputs = normalize_output_tensors(outputs, config.resolved_output_names)
    if config.schema == YOLO_POSE_RAW:
        return _decode_yolo_raw(named_outputs, config, inverse_affine)
    if config.schema in (YOLO_POSE_NMS, YOLO_POSE_END2END):
        return _decode_yolo_processed(named_outputs, config, inverse_affine)
    if config.schema == RTMO_MMDEPLOY:
        return _decode_rtmo_mmdeploy(named_outputs, config, inverse_affine)
    if config.schema == RTMPOSE_SIMCC:
        return [decode_simcc(named_outputs, config, inverse_affine)]
    raise ValueError(f"No decoder registered for output schema {config.schema!r}")


__all__ = [
    "PoseInstance",
    "PoseOutputConfig",
    "SUPPORTED_OUTPUT_SCHEMAS",
    "YOLO_POSE_RAW",
    "YOLO_POSE_NMS",
    "YOLO_POSE_END2END",
    "RTMO_MMDEPLOY",
    "RTMPOSE_SIMCC",
    "normalize_output_tensors",
    "decode_simcc",
    "decode_pose_outputs",
]
