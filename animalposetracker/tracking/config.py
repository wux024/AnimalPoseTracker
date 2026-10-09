"""Configuration for independent keypoint-first online trackers."""

from dataclasses import dataclass, fields
import math
from typing import Any, Dict, Mapping, Optional, Tuple


ALGORITHMS = (
    "bytetrack",
    "botsort",
    "ocsort",
    "deepocsort",
    "fasttrack",
    "tracktrack",
)

_DEFAULTS = {
    # Values follow the currently published tracker YAMLs where applicable. The
    # implementation and configuration remain local to AnimalPoseTracker.
    "bytetrack": {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
        "min_track_len": 1,
        "gmc_method": "none",
    },
    "botsort": {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
        "min_track_len": 1,
        "gmc_method": "none",
    },
    "ocsort": {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
        "delta_t": 3,
        "inertia": 0.20,
        "use_byte": False,
        "min_track_len": 1,
        "gmc_method": "none",
    },
    "deepocsort": {
        "track_high_thresh": 0.30,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.30,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
        "delta_t": 3,
        "inertia": 0.20,
        "use_byte": False,
        "gmc_method": "none",
        "min_track_len": 1,
        "appearance_thresh": 0.90,
    },
    "fasttrack": {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
        "reset_velocity_offset_occ": 5,
        "reset_pos_offset_occ": 3,
        "enlarge_bbox_occ": 1.10,
        "dampen_motion_occ": 0.50,
        "active_occ_to_lost_thresh": 10,
        "occ_cover_thresh": 0.70,
        "occ_reappear_window": 40,
        "init_iou_suppress": 0.70,
        "init_pose_suppress": 0.85,
        "min_track_len": 1,
        "gmc_method": "none",
    },
    "tracktrack": {
        "track_high_thresh": 0.60,
        "track_low_thresh": 0.25,
        "new_track_thresh": 0.70,
        "track_buffer": 30,
        "match_thresh": 0.70,
        "lost_match_thr": 0.0,
        "reid_weight": 0.50,
        "conf_weight": 0.10,
        "angle_weight": 0.05,
        "penalty_p": 0.20,
        "penalty_q": 0.40,
        "reduce_step": 0.05,
        "tai_oks_thr": 0.85,
        "min_track_len": 3,
        "gmc_method": "none",
    },
}

_REID_ALGORITHMS = {"botsort", "deepocsort", "tracktrack"}
_GMC_ALGORITHMS = {"botsort", "deepocsort", "tracktrack"}
_GMC_METHODS = {"none", "sparseOptFlow", "orb", "sift", "ecc"}


@dataclass
class TrackerConfig:
    """Tunable keypoint-first tracker parameters; ByteTrack is the default strategy."""

    algorithm: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    min_track_len: int = 1

    with_reid: bool = False
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.80
    alpha_fixed_emb: float = 0.95

    gmc_method: str = "none"

    delta_t: int = 3
    inertia: float = 0.20
    use_byte: bool = False

    reset_velocity_offset_occ: int = 5
    reset_pos_offset_occ: int = 3
    enlarge_bbox_occ: float = 1.10
    dampen_motion_occ: float = 0.50
    active_occ_to_lost_thresh: int = 10
    occ_cover_thresh: float = 0.70
    occ_reappear_window: int = 40
    init_iou_suppress: float = 0.70
    init_pose_suppress: float = 0.85

    lost_match_thr: float = 0.0
    pose_weight: float = 1.0
    box_weight: float = 0.0
    reid_weight: float = 0.50
    conf_weight: float = 0.10
    angle_weight: float = 0.05
    penalty_p: float = 0.20
    penalty_q: float = 0.40
    reduce_step: float = 0.05
    tai_oks_thr: float = 0.85

    pose_keypoint_thresh: float = 0.25
    pose_min_common_keypoints: int = 3
    kpt_oks_sigmas: Optional[Tuple[float, ...]] = None

    @classmethod
    def for_algorithm(cls, algorithm: str = "bytetrack", **overrides):
        values = dict(_DEFAULTS.get(str(algorithm).lower(), {}))
        values.update(overrides)
        values["algorithm"] = str(algorithm).lower()
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        algorithm = str(values.get("algorithm", values.get("tracker_type", "bytetrack"))).lower()
        if algorithm not in _DEFAULTS:
            raise ValueError(f"Unknown tracker algorithm {algorithm!r}; choose one of {ALGORITHMS}")
        merged = dict(_DEFAULTS[algorithm])
        merged.update(values)
        merged["algorithm"] = algorithm

        valid = {item.name for item in fields(cls)}
        unknown = sorted(set(merged) - valid - {"tracker_type"})
        if unknown:
            raise ValueError(f"Unknown tracker configuration fields: {unknown}")
        merged.pop("tracker_type", None)
        config = cls(**merged)
        config.validate()
        return config

    def validate(self):
        self.algorithm = str(self.algorithm).lower()
        if self.algorithm not in _DEFAULTS:
            raise ValueError(f"Unknown tracker algorithm {self.algorithm!r}; choose one of {ALGORITHMS}")
        for name in (
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "match_thresh",
            "proximity_thresh",
            "appearance_thresh",
            "dampen_motion_occ",
            "occ_cover_thresh",
            "init_iou_suppress",
            "init_pose_suppress",
            "lost_match_thr",
            "pose_weight",
            "box_weight",
            "reid_weight",
            "conf_weight",
            "angle_weight",
            "penalty_p",
            "penalty_q",
            "reduce_step",
            "tai_oks_thr",
            "pose_keypoint_thresh",
            "alpha_fixed_emb",
            "inertia",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh cannot exceed track_high_thresh")
        if self.pose_weight <= 0.0 and self.box_weight <= 0.0:
            raise ValueError("pose_weight or box_weight must be positive")
        if self.track_buffer < 0 or self.min_track_len < 1:
            raise ValueError("track_buffer must be non-negative and min_track_len at least 1")
        if self.delta_t < 1:
            raise ValueError("delta_t must be at least 1")
        if self.pose_min_common_keypoints < 1:
            raise ValueError("pose_min_common_keypoints must be at least 1")
        if self.kpt_oks_sigmas is not None:
            sigmas = tuple(float(value) for value in self.kpt_oks_sigmas)
            if not sigmas or any(not math.isfinite(value) or value <= 0.0 for value in sigmas):
                raise ValueError("kpt_oks_sigmas must contain finite positive values")
            self.kpt_oks_sigmas = sigmas
        if self.reset_velocity_offset_occ < 0 or self.reset_pos_offset_occ < 0:
            raise ValueError("occlusion rollback offsets must be non-negative")
        if self.active_occ_to_lost_thresh < 0 or self.occ_reappear_window < 0:
            raise ValueError("occlusion frame thresholds must be non-negative")
        if not math.isfinite(float(self.enlarge_bbox_occ)) or self.enlarge_bbox_occ < 1.0:
            raise ValueError("enlarge_bbox_occ must be finite and at least 1")
        if self.with_reid and self.algorithm not in _REID_ALGORITHMS:
            raise ValueError(f"{self.algorithm} does not define a ReID association branch")
        if self.gmc_method not in _GMC_METHODS:
            raise ValueError(f"gmc_method must be one of {sorted(_GMC_METHODS)}")
        if self.gmc_method != "none" and self.algorithm not in _GMC_ALGORITHMS:
            raise ValueError(f"{self.algorithm} does not define camera motion compensation")
        return self


__all__ = ["TrackerConfig", "ALGORITHMS"]
