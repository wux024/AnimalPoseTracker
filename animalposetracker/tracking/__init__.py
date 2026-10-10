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
from .base import BaseTracker
from .config import ALGORITHMS, TrackerConfig
from .types import DetectionInput, DetectionSequence, PoseDetection, TrackedDetection

__all__ = [
    "ALGORITHMS",
    "BaseTracker",
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
