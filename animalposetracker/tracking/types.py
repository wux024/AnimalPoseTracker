"""Public, model-agnostic observation and track result types."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _finite_array(value, name: str, ndim: int):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return np.ascontiguousarray(array.copy())


@dataclass(frozen=True)
class PoseDetection:
    """One frame's detector/pose observation, independent of model implementation.

    A box is optional; if omitted, the tracker derives a loose internal extent
    from visible keypoints. When keypoints are present, they drive association.
    ``keypoints`` is ``(K, 2)`` or ``(K, 3)`` with the optional third column
    carrying confidence/visibility. ``embedding`` is an optional precomputed
    appearance vector; the tracker never loads or trains an encoder.
    """

    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    score: Optional[float] = None
    class_id: int = 0
    keypoints: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        box = None
        if self.bbox_xyxy is not None:
            box = np.asarray(self.bbox_xyxy, dtype=np.float32).reshape(-1)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("bbox_xyxy must contain four finite coordinates when provided")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("bbox_xyxy must have positive width and height")
            object.__setattr__(self, "bbox_xyxy", tuple(float(item) for item in box))
        if self.score is None:
            raise ValueError("score is required")
        score = float(self.score)
        if not np.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "class_id", int(self.class_id))

        if self.keypoints is not None:
            points = np.asarray(self.keypoints, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] not in (2, 3):
                raise ValueError("keypoints must have shape (K, 2) or (K, 3)")
            points = np.ascontiguousarray(points.copy())
            if points.shape[1] == 3:
                if not np.isfinite(points[:, 2]).all() or np.any((points[:, 2] < 0.0) | (points[:, 2] > 1.0)):
                    raise ValueError("keypoint confidence/visibility values must be finite and in [0, 1]")
                visible = points[:, 2] > 0.0
                if not np.isfinite(points[visible, :2]).all():
                    raise ValueError("visible keypoint coordinates must be finite")
                points[~visible, :2] = 0.0
            elif not np.isfinite(points).all():
                raise ValueError("2-D keypoint coordinates must be finite")
            if len(points) == 0:
                raise ValueError("keypoints must contain at least one joint")
            object.__setattr__(self, "keypoints", points)

        if self.bbox_xyxy is None and self.keypoints is None:
            raise ValueError("at least one of bbox_xyxy or keypoints is required")
        if (
            self.bbox_xyxy is None
            and self.keypoints is not None
            and self.keypoints.shape[1] == 3
            and not np.any(self.keypoints[:, 2] > 0.0)
        ):
            raise ValueError("a boxless pose detection must contain at least one visible keypoint")

        if self.embedding is not None:
            embedding = _finite_array(self.embedding, "embedding", 1)
            if embedding.size == 0:
                raise ValueError("embedding must not be empty")
            norm = float(np.linalg.norm(embedding))
            if norm > 0.0:
                embedding /= norm
            object.__setattr__(self, "embedding", embedding)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PoseDetection":
        """Construct an observation from a plain mapping without importing a predictor."""
        if not isinstance(value, Mapping):
            raise TypeError("A detection must be PoseDetection or a mapping")
        return cls(
            bbox_xyxy=value.get("bbox_xyxy"),
            score=value["score"],
            class_id=value.get("class_id", 0),
            keypoints=value.get("keypoints"),
            embedding=value.get("embedding"),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": None if self.bbox_xyxy is None else list(self.bbox_xyxy),
            "score": self.score,
            "class_id": self.class_id,
            "keypoints": None if self.keypoints is None else self.keypoints.copy(),
            "embedding": None if self.embedding is None else self.embedding.copy(),
        }


@dataclass(frozen=True)
class TrackedDetection:
    """A current-frame observation with its sequence-local identity and lifecycle."""

    detection: PoseDetection
    track_id: int
    age: int
    hits: int
    is_confirmed: bool

    @property
    def bbox_xyxy(self):
        return self.detection.bbox_xyxy

    @property
    def score(self):
        return self.detection.score

    @property
    def class_id(self):
        return self.detection.class_id

    @property
    def keypoints(self):
        return self.detection.keypoints

    @property
    def embedding(self):
        return self.detection.embedding

    def to_mapping(self) -> Dict[str, Any]:
        result = self.detection.to_mapping()
        result.update({
            "track_id": int(self.track_id),
            "age": int(self.age),
            "hits": int(self.hits),
            "is_confirmed": bool(self.is_confirmed),
        })
        return result


DetectionInput = PoseDetection
DetectionSequence = Sequence[PoseDetection]


__all__ = ["PoseDetection", "TrackedDetection", "DetectionInput", "DetectionSequence"]
