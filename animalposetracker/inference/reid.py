"""Appearance encoders for the ReID-assisted trackers.

Lives in the inference layer because it owns a runtime: ``tracking/`` stays pure
algorithm and only ever reads ``PoseDetection.embedding``.

One source: ``OnnxReIDEncoder``, an exported model (``yolo26*-reid.onnx`` or one trained
on the target species), one crop-sized forward per box. Reusing the detector's own
features used to be an option; it required guessing where the backbone ends, and a wrong
guess produced plausible-looking but useless embeddings, so it was dropped rather than
guarded.

The encoder returns one L2-normalised vector per detection, so the association code only
needs cosine similarity and never learns which model produced the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

__all__ = [
    "NullReIDEncoder",
    "OnnxReIDEncoder",
    "ReIDEncoder",
    "attach_embeddings",
    "build_reid_encoder",
]


class ReIDEncoder(Protocol):
    """Turns image crops into appearance embeddings."""

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Return an ``(N, D)`` array of L2-normalised embeddings, one row per box.

        Args:
            frame: The BGR frame the boxes were detected in.
            boxes: ``(N, 4)`` integer ``x1, y1, x2, y2`` boxes.
        """


class NullReIDEncoder:
    """Refuses to encode.

    ``build_reid_encoder`` returns this when ``with_reid`` is on but no model is set, so
    a misconfiguration fails at the first association instead of quietly scoring every
    pair as dissimilar.
    """

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "with_reid is enabled but no ReID model is configured; "
            "set the tracker model to a .onnx path, or turn with_reid off"
        )


class OnnxReIDEncoder:
    """Runs an exported ReID model on cropped boxes.

    ``backend`` mirrors the pose estimator's two runtimes (``"ort"`` / ``"cv2"``) so a
    ReID run never lands on a different engine than the detector that produced its crops.
    """

    def __init__(
        self,
        model_path,
        imgsz: int = 224,
        providers: Sequence[str] | None = None,
        backend: str = "ort",
        device: str = "CPU",
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ReID model not found: {path}")
        if int(imgsz) <= 0:
            raise ValueError("imgsz must be positive")
        if backend not in {"ort", "cv2"}:
            raise ValueError("backend must be 'ort' or 'cv2'")
        self.imgsz = int(imgsz)
        self.backend = backend
        self.device = device

        if backend == "cv2":
            import cv2

            from .constant import ENGINEtoBackend, OpenCV_TARGETS

            self.net = cv2.dnn.readNetFromONNX(str(path))
            self.net.setPreferableBackend(
                ENGINEtoBackend["OpenCV"].get(device, cv2.dnn.DNN_BACKEND_OPENCV)
            )
            self.net.setPreferableTarget(OpenCV_TARGETS.get(device, cv2.dnn.DNN_TARGET_CPU))
            self.input_height = self.input_width = self.imgsz
            return

        import onnxruntime as ort  # local import keeps this module import-light

        from .detector_adapter import _default_providers

        self.providers = (
            list(providers) if providers else _default_providers(ort.get_available_providers())
        )
        self.session = ort.InferenceSession(str(path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        # Static shapes are the norm for ReID exports; fall back to the requested square
        # only when the model declares a dynamic spatial axis.
        shape = self.session.get_inputs()[0].shape
        height = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else self.imgsz
        width = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else self.imgsz
        self.input_height = int(height)
        self.input_width = int(width)

    def _crop(self, frame: np.ndarray, box: Sequence[int]) -> np.ndarray:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        return crop.astype(np.float32) / 255.0

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        batch = np.stack([self._crop(frame, box) for box in boxes], axis=0)
        # NCHW with BGR -> RGB. ReID exports consumed here take 0..1 input; the channel
        # swap sits in one place instead of at every call site.
        blob = batch[:, :, :, ::-1].transpose(0, 3, 1, 2).copy()
        if self.backend == "cv2":
            self.net.setInput(blob)
            output = self.net.forward()
        else:
            output = self.session.run(None, {self.input_name: blob})[0]
        embeddings = np.asarray(output, dtype=np.float32).reshape(len(boxes), -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms


def build_reid_encoder(kind=None, providers=None, backend: str = "ort", device: str = "CPU") -> ReIDEncoder | None:
    """Resolve ``with_reid`` / ``model`` settings into an encoder.

    Args:
        kind: ``None`` or ``"none"`` disables ReID; anything else is a model path.
        providers: onnxruntime providers; ``None`` picks the best this build offers.
        backend: ``"ort"`` or ``"cv2"``, matching whatever the pose estimator uses.
        device: Device label shared with the pose estimator's OpenCV backend table.

    Returns ``None`` when ReID is off, so callers skip the appearance stage entirely
    instead of paying for an encoder that refuses to run.
    """
    if kind is None or str(kind).lower() in {"", "none", "off", "false", "0"}:
        return None
    return OnnxReIDEncoder(kind, providers=providers, backend=backend, device=device)


def attach_embeddings(records, frame, encoder):
    """Return ``records`` with one appearance vector attached per detection.

    Shared by the stream pipeline and the offline video writer: both hold records as
    plain mappings at this point, and both must hand the tracker vectors that were cut
    from the same frame the boxes were detected in. A record with neither a box nor
    usable keypoints is left without an embedding, which the tracker refuses rather than
    silently scoring.
    """
    if encoder is None or not records:
        return records
    from animalposetracker.tracking.geometry import keypoints_to_bbox

    boxes, indices = [], []
    for index, record in enumerate(records):
        box = record.get("bbox_xyxy")
        if box is None:
            points = record.get("keypoints")
            box = keypoints_to_bbox(points) if points is not None else None
        if box is None:
            continue
        boxes.append(np.asarray(box, dtype=np.float32))
        indices.append(index)
    if not boxes:
        return records
    vectors = encoder.encode(frame, np.asarray(boxes, dtype=np.float32))
    for index, vector in zip(indices, vectors):
        records[index] = {**records[index], "embedding": vector}
    return records
