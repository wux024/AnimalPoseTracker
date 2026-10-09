"""Composable model-output decoding for live stream inference.

This API consumes in-memory frames only. File and dataset prediction remain in
``predict_cli`` and are intentionally not imported here.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import cv2
import numpy as np

from animalposetracker.preprocessing.topdown import (
    _rotate_point as _rotate_point_radians,
    _third_point,
    _topdown_warp_matrix as _topdown_warp_matrix_with_rotation,
    _fix_aspect_ratio,
)
from animalposetracker.postprocessing.output_decoders import (
    RTMO_MMDEPLOY,
    RTMPOSE_SIMCC,
    PoseInstance,
    PoseOutputConfig,
    decode_pose_outputs,
    decode_simcc,
)


@dataclass(frozen=True)
class DetectorBox:
    """A detector result in the original stream-frame coordinate system."""

    bbox_xyxy: np.ndarray
    score: float = 1.0
    class_id: Optional[int] = None

    def __post_init__(self):
        box = np.asarray(self.bbox_xyxy, dtype=np.float32).reshape(4)
        if not np.isfinite(box).all():
            raise ValueError("detector bbox_xyxy must contain finite coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("detector bbox_xyxy must have positive width and height")
        if not np.isfinite(float(self.score)):
            raise ValueError("detector score must be finite")
        object.__setattr__(self, "bbox_xyxy", box)


class DetectorProvider(Protocol):
    """Injection point for a separately configured stream detector model.

    Implementations own the detector weights/backend and return final boxes in
    original-frame coordinates. This package does not select a detector family.
    """

    def predict_boxes(self, frame: np.ndarray) -> Sequence[DetectorBox]:
        """Return thresholded, postprocessed boxes for one in-memory frame."""


@dataclass(frozen=True)
class StreamInputConfig:
    """Preprocessing metadata for an exported stream model."""

    input_size: Optional[Tuple[int, int]] = None  # width, height
    preprocess_mode: Optional[str] = None  # engine | mmpose_bottomup
    color_order: str = "RGB"
    input_scale: float = 1.0
    pixel_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pixel_std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    pad_value: float = 114.0
    bbox_padding: float = 1.25

    def __post_init__(self):
        if self.input_size is not None:
            object.__setattr__(
                self, "input_size", tuple(map(int, self.input_size))
            )
        object.__setattr__(self, "pixel_mean", tuple(self.pixel_mean))
        object.__setattr__(self, "pixel_std", tuple(self.pixel_std))
        if self.input_size is not None:
            width, height = self.input_size
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError("input_size must contain positive width and height")
        if self.preprocess_mode not in (None, "engine", "mmpose_bottomup"):
            raise ValueError(
                "preprocess_mode must be 'engine' or 'mmpose_bottomup'"
            )
        if self.color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must be 'RGB' or 'BGR'")
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive")
        if len(self.pixel_mean) != 3 or len(self.pixel_std) != 3:
            raise ValueError("pixel_mean and pixel_std must have three channels")
        if any(float(value) == 0.0 for value in self.pixel_std):
            raise ValueError("pixel_std values must be non-zero")
        if self.bbox_padding <= 0:
            raise ValueError("bbox_padding must be positive")


def _rotate_point(point: np.ndarray, angle: float) -> np.ndarray:
    """Compatibility helper whose public angle is expressed in degrees."""
    return _rotate_point_radians(point, np.deg2rad(float(angle)))


def _topdown_warp_matrix(center, scale, output_size) -> np.ndarray:
    return _topdown_warp_matrix_with_rotation(center, scale, 0.0, output_size)


def _to_model_tensor(image: np.ndarray, config: StreamInputConfig) -> np.ndarray:
    if config.color_order == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) * float(config.input_scale)
    mean = np.asarray(config.pixel_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config.pixel_std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def _preprocess_mmpose_bottomup(
    frame: np.ndarray,
    config: StreamInputConfig,
    fallback_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = config.input_size or fallback_size
    image_height, image_width = frame.shape[:2]
    center = np.asarray([image_width * 0.5, image_height * 0.5], dtype=np.float32)
    scale = _fix_aspect_ratio(
        np.asarray([image_width, image_height], dtype=np.float32), (width, height)
    )
    warp = _topdown_warp_matrix(center, scale, (width, height))
    resized = cv2.warpAffine(
        frame,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(config.pad_value,) * 3,
    )
    inverse = cv2.invertAffineTransform(warp).astype(np.float32)
    return _to_model_tensor(resized, config), inverse


def _preprocess_topdown_crop(
    frame: np.ndarray,
    detection: DetectorBox,
    config: StreamInputConfig,
    fallback_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if config.input_size is None:
        raise ValueError(
            "RTMPose top-down inference requires an explicit input_size (width, height)"
        )
    width, height = config.input_size or fallback_size
    box = detection.bbox_xyxy
    center = (box[:2] + box[2:]) * 0.5
    scale = (box[2:] - box[:2]) * float(config.bbox_padding)
    scale = _fix_aspect_ratio(scale, (width, height))
    warp = _topdown_warp_matrix(center, scale, (width, height))
    crop = cv2.warpAffine(
        frame,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    inverse = cv2.invertAffineTransform(warp).astype(np.float32)
    return _to_model_tensor(crop, config), inverse


def _coerce_detector_box(value: Any) -> DetectorBox:
    if isinstance(value, DetectorBox):
        return value
    if isinstance(value, Mapping):
        if "bbox_xyxy" not in value:
            raise ValueError("detector results must include bbox_xyxy")
        return DetectorBox(
            bbox_xyxy=value["bbox_xyxy"],
            score=float(value.get("score", 1.0)),
            class_id=(
                int(value["class_id"]) if value.get("class_id") is not None else None
            ),
        )
    raise TypeError(
        "detector results must be DetectorBox objects or mappings with bbox_xyxy"
    )


class StreamInferencePipeline:
    """Run one-stage or detector-plus-top-down pose inference on stream frames.

    ``pose_engine`` is an initialized :class:`InferenceEngine`. For RTMPose,
    callers inject a separate detector adapter with its own weights/runtime via
    ``detector``. The adapter must return boxes in source-frame coordinates.
    """

    def __init__(
        self,
        pose_engine,
        output_config: Union[PoseOutputConfig, Mapping[str, Any]],
        input_mode: Optional[str] = None,
        input_config: Optional[Union[StreamInputConfig, Mapping[str, Any]]] = None,
        detector: Optional[DetectorProvider] = None,
    ):
        if pose_engine is None:
            raise ValueError("pose_engine is required")
        if isinstance(output_config, Mapping):
            output_config = PoseOutputConfig(**output_config)
        if input_config is not None and isinstance(input_config, Mapping):
            input_config = StreamInputConfig(**input_config)
        if not isinstance(output_config, PoseOutputConfig):
            raise TypeError("output_config must be PoseOutputConfig or a mapping")
        if input_config is not None and not isinstance(input_config, StreamInputConfig):
            raise TypeError("input_config must be StreamInputConfig or a mapping")
        if input_mode is None:
            input_mode = "topdown" if output_config.schema == RTMPOSE_SIMCC else "full_frame"
        if input_mode not in ("full_frame", "topdown"):
            raise ValueError("input_mode must be 'full_frame' or 'topdown'")
        if output_config.schema == RTMPOSE_SIMCC and input_mode != "topdown":
            raise ValueError("RTMPose SimCC must use input_mode='topdown'")
        if input_mode == "topdown" and output_config.schema != RTMPOSE_SIMCC:
            raise ValueError(
                "The top-down stream pipeline currently supports RTMPose SimCC outputs"
            )
        if input_mode == "topdown" and detector is None:
            raise ValueError(
                "Top-down stream inference requires a separate detector provider "
                "and detector model"
            )
        if detector is not None and not callable(getattr(detector, "predict_boxes", None)):
            raise TypeError("detector must implement predict_boxes(frame)")
        if input_config is None:
            if output_config.schema == RTMPOSE_SIMCC:
                input_config = StreamInputConfig(
                    color_order="RGB",
                    input_scale=1.0,
                    pixel_mean=(123.675, 116.28, 103.53),
                    pixel_std=(58.395, 57.12, 57.375),
                )
            elif output_config.schema == RTMO_MMDEPLOY:
                input_config = StreamInputConfig(
                    preprocess_mode="mmpose_bottomup",
                    color_order="BGR",
                    input_scale=1.0,
                    pixel_mean=(0.0, 0.0, 0.0),
                    pixel_std=(1.0, 1.0, 1.0),
                )
            else:
                input_config = StreamInputConfig(preprocess_mode="engine")
        if input_mode == "topdown" and input_config.input_size is None:
            raise ValueError(
                "Top-down stream inference requires input_config.input_size"
            )
        if input_config.preprocess_mode == "mmpose_bottomup" and output_config.schema != RTMO_MMDEPLOY:
            raise ValueError("mmpose_bottomup preprocessing is reserved for RTMO outputs")

        self.pose_engine = pose_engine
        self.output_config = output_config
        self.input_mode = input_mode
        self.input_config = input_config
        self.detector = detector

    def _pose_input_size(self) -> Tuple[int, int]:
        if self.input_config.input_size is not None:
            return tuple(map(int, self.input_config.input_size))
        return int(self.pose_engine.input_width), int(self.pose_engine.input_height)

    def _inference_outputs(self, tensor):
        if self.pose_engine.model is None:
            raise RuntimeError("Pose engine is not loaded; call model_init() first.")
        return self.pose_engine.inference_named_outputs(
            tensor,
            output_names=self.pose_engine.output_names or None,
        )

    def _predict_full_frame(self, frame: np.ndarray) -> List[PoseInstance]:
        mode = self.input_config.preprocess_mode or "engine"
        if mode == "engine":
            model_input, inverse = self.pose_engine.preprocess(frame)
        elif mode == "mmpose_bottomup":
            model_input, inverse = _preprocess_mmpose_bottomup(
                frame, self.input_config, self._pose_input_size()
            )
        else:
            raise ValueError(f"Unsupported full-frame preprocess mode {mode!r}")
        outputs = self._inference_outputs(model_input)
        return decode_pose_outputs(outputs, self.output_config, inverse_affine=inverse)

    def _predict_topdown(self, frame: np.ndarray) -> List[PoseInstance]:
        boxes = [
            _coerce_detector_box(box)
            for box in self.detector.predict_boxes(frame)
        ]
        instances = []
        for detection in boxes:
            model_input, inverse = _preprocess_topdown_crop(
                frame,
                detection,
                self.input_config,
                self._pose_input_size(),
            )
            outputs = self._inference_outputs(model_input)
            pose = decode_simcc(
                outputs,
                self.output_config,
                inverse_affine=inverse,
                bbox_xyxy=detection.bbox_xyxy,
                bbox_score=detection.score,
                class_id=detection.class_id,
            )
            instances.append(pose)
        return instances

    def process_frame(self, frame: np.ndarray) -> List[PoseInstance]:
        """Consume one in-memory BGR stream frame and return decoded instances."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 NumPy array")
        if self.input_mode == "topdown":
            return self._predict_topdown(frame)
        return self._predict_full_frame(frame)


__all__ = [
    "DetectorBox",
    "DetectorProvider",
    "StreamInputConfig",
    "StreamInferencePipeline",
]
