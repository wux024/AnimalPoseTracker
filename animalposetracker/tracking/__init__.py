"""Standalone animal detection and pose tracking algorithms."""

from .algorithms import (
    BoTSORT,
    ByteTrack,
    DeepOCSORT,
    FastTracker,
    OCSORT,
    TrackTrack,
    create_tracker,
)
from .config import ALGORITHMS, TrackerConfig
from .types import DetectionInput, DetectionSequence, PoseDetection, TrackedDetection

__all__ = [
    "ALGORITHMS",
    "TrackerConfig",
    "PoseDetection",
    "TrackedDetection",
    "DetectionInput",
    "DetectionSequence",
    "ByteTrack",
    "BoTSORT",
    "OCSORT",
    "DeepOCSORT",
    "FastTracker",
    "TrackTrack",
    "create_tracker",
]
