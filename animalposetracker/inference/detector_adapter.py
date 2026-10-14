"""ONNX detector adapters for the top-down stream pipeline.

A thin, framework-decoupled implementation of ``DetectorProvider`` (see
``stream_inference``): it feeds a classic real-time detector exported to ONNX
and returns thresholded boxes in source-frame coordinates. Nothing in this
module imports a training framework -- only onnxruntime, numpy and OpenCV.

Contract: any detector that can be described as

* output shape ``(1, N, 6)``, in ``[cx, cy, w, h, score, class]`` order for
  ``box_format="cxcywh"`` (RT-DETR exports) or ``[x1, y1, x2, y2, score,
  class]`` for ``box_format="xyxy"`` (the NMS-free YOLO exports, e.g. YOLO26 /
  YOLOv10 exported with ``nms=False``),
* a coordinate convention (``normalized`` like RT-DETR, or ``pixels`` of the
  network input like the YOLO family), and
* a preprocessing convention (``stretch`` resize, or ``letterbox`` resize with
  padding),

plugs in through :class:`OnnxDetectorAdapter` with those declarations. The
defaults reproduce the RT-DETR stream export (verified against the official
postprocess: stretch-resize preprocessing, normalized coordinates, multiply by
the source size, no NMS on the way out).

The ONNX session prefers ``CUDAExecutionProvider`` whenever the installed
onnxruntime build offers it (``onnxruntime-gpu``) and falls back to plain CPU
otherwise; ``providers`` overrides both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from animalposetracker.inference.stream_inference import DetectorBox

_COORD_MODES = ("normalized", "pixels")
_PREPROCESS_MODES = ("stretch", "letterbox")
_BOX_FORMATS = ("cxcywh", "xyxy")

#: Session provider preference: the GPU when this onnxruntime build has one,
#: plain CPU otherwise. A two-stage stream on a CPU-only build spends most of
#: the frame in the detector, so getting this right is what makes a server run
#: fast.
_PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def _default_providers(available) -> list:
    """Keep the preferred providers this onnxruntime build actually offers."""
    offered = set(available or ())
    chosen = [name for name in _PREFERRED_PROVIDERS if name in offered]
    return chosen or ["CPUExecutionProvider"]


class OnnxDetectorAdapter:
    """(1, N, 6) ONNX detector with declarable coordinate/preprocess conventions."""

    def __init__(
        self,
        onnx_path,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.5,
        class_ids: Optional[Sequence[int]] = (0,),  # person by default
        max_det: int = 300,
        providers: Optional[Sequence[str]] = None,
        coord_mode: str = "normalized",
        preprocess_mode: str = "stretch",
        box_format: str = "cxcywh",
    ):
        import onnxruntime as ort  # local import keeps this module import-light

        if int(input_size) <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 <= float(conf_threshold) <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if coord_mode not in _COORD_MODES:
            raise ValueError(f"coord_mode must be one of {_COORD_MODES}")
        if preprocess_mode not in _PREPROCESS_MODES:
            raise ValueError(f"preprocess_mode must be one of {_PREPROCESS_MODES}")
        if box_format not in _BOX_FORMATS:
            raise ValueError(f"box_format must be one of {_BOX_FORMATS}")
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"detector ONNX not found: {self.onnx_path}")
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.class_ids = tuple(int(value) for value in class_ids) if class_ids else None
        self.max_det = int(max_det)
        self.coord_mode = coord_mode
        self.preprocess_mode = preprocess_mode
        self.box_format = box_format
        self.providers = (
            list(providers) if providers else _default_providers(ort.get_available_providers())
        )
        self.session = ort.InferenceSession(str(self.onnx_path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------ #
    # preprocessing                                                      #
    # ------------------------------------------------------------------ #
    def _prepare_canvas(self, frame: np.ndarray):
        """Return (network image uint8, pad_x, pad_y, ratio_x, ratio_y).

        ``ratio_*`` maps source-frame pixels to network-canvas pixels, so the
        inverse mapping is ``source = (canvas - pad) / ratio``.
        """
        height, width = frame.shape[:2]
        if self.preprocess_mode == "letterbox":
            scale = min(self.input_size / width, self.input_size / height)
            new_w, new_h = round(width * scale), round(height * scale)
            canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
            pad_x, pad_y = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
            resized = cv2.resize(frame, (new_w, new_h))
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            return canvas, float(pad_x), float(pad_y), scale, scale
        # stretch: independent per-axis scaling, no padding
        canvas = cv2.resize(frame, (self.input_size, self.input_size))
        return canvas, 0.0, 0.0, self.input_size / width, self.input_size / height

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR->RGB, /255, NCHW tensor for the network."""
        canvas, *_ = self._prepare_canvas(frame)
        image = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return image.transpose(2, 0, 1)[None]

    # ------------------------------------------------------------------ #
    # detection                                                          #
    # ------------------------------------------------------------------ #
    def predict_boxes(self, frame: np.ndarray) -> Sequence[DetectorBox]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 NumPy array")
        canvas, pad_x, pad_y, ratio_x, ratio_y = self._prepare_canvas(frame)
        tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0
        outputs = self.session.run(None, {self.input_name: tensor.transpose(2, 0, 1)[None]})
        rows = np.asarray(outputs[0])
        if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[2] != 6:
            raise ValueError(
                f"expected a (1, N, 6) [cx, cy, w, h, score, class] output, got {rows.shape}"
            )
        rows = rows[0]
        keep = rows[:, 4] >= self.conf_threshold
        if self.class_ids is not None:
            keep &= np.isin(np.rint(rows[:, 5]).astype(int), np.asarray(self.class_ids))
        kept = rows[keep][: self.max_det]
        height, width = frame.shape[:2]
        scale = float(self.input_size) if self.coord_mode == "normalized" else 1.0
        boxes = []
        for a, b, c, d, score, class_id in kept:
            # 1) to network-canvas corner form, 2) undo padding/ratio -> source frame
            if self.box_format == "xyxy":
                x1c, y1c, x2c, y2c = a, b, c, d
            else:  # cxcywh
                x1c, y1c = a - c * 0.5, b - d * 0.5
                x2c, y2c = a + c * 0.5, b + d * 0.5
            x1 = (x1c * scale - pad_x) / ratio_x
            y1 = (y1c * scale - pad_y) / ratio_y
            x2 = (x2c * scale - pad_x) / ratio_x
            y2 = (y2c * scale - pad_y) / ratio_y
            x1, y1 = max(x1, 0.0), max(y1, 0.0)
            x2, y2 = min(x2, float(width)), min(y2, float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(
                DetectorBox(
                    bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    score=float(score),
                    class_id=int(round(float(class_id))),
                )
            )
        return boxes


__all__ = ["OnnxDetectorAdapter"]
