"""Independent online pose association algorithms.

The implementations share the local Kalman/lifecycle primitives, but keep each
tracker's association policy separate so callers can select one without pulling
in a model, GUI, or inference runtime.
"""

from collections import deque
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .base import BaseTracker, Track
from .config import TrackerConfig
from .geometry import bbox_iou, pose_similarity
from .types import PoseDetection


def _score_indices(detections: Sequence[PoseDetection], predicate) -> List[int]:
    return [index for index, detection in enumerate(detections) if predicate(detection.score)]


def _unique(values: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(values))


class ByteTrack(BaseTracker):
    """Two-stage high/low confidence recovery with OKS-based pose association."""

    def _associate(self, detections: Sequence[PoseDetection]):
        cfg = self.config
        track_ids = [track.track_id for track in self.tracks.values() if track.state != "removed"]
        high = _score_indices(detections, lambda score: score >= cfg.track_high_thresh)
        low = _score_indices(
            detections,
            lambda score: cfg.track_low_thresh <= score < cfg.track_high_thresh,
        )

        confirmed_ids = [
            track_id for track_id in track_ids
            if self.tracks[track_id].state in {"tracked", "lost"}
        ]
        tentative_ids = [
            track_id for track_id in track_ids if self.tracks[track_id].state == "tentative"
        ]
        matches, unmatched_confirmed, unmatched_high = self._assignment(
            confirmed_ids, high, detections, mode="pose"
        )

        # ByteTrack's second association rescues active tracks with low-score poses.
        active_unmatched = [
            track_id for track_id in unmatched_confirmed
            if self.tracks[track_id].state == "tracked"
        ]
        low_matches, still_unmatched_active, unmatched_low = self._assignment(
            active_unmatched, low, detections, mode="low", max_cost=0.5
        )
        matches.extend(low_matches)

        tentative_matches, unmatched_tentative, unmatched_high = self._assignment(
            tentative_ids, unmatched_high, detections, mode="pose", max_cost=0.7
        )
        matches.extend(tentative_matches)

        unmatched_tracks = _unique(
            [track_id for track_id in unmatched_confirmed if track_id not in active_unmatched]
            + still_unmatched_active
            + unmatched_tentative
        )
        unmatched_detections = _unique(unmatched_high + unmatched_low)
        return matches, unmatched_tracks, unmatched_detections


class BoTSORT(ByteTrack):
    """ByteTrack-style pose matching with optional camera and appearance cues.

    GMC is applied by the shared lifecycle before association. ReID embeddings
    participate only when ``with_reid`` is explicitly enabled.
    """


class OCSORT(BaseTracker):
    """Observation-centric keypoint motion matching and recovery."""

    def _associate(self, detections: Sequence[PoseDetection]):
        cfg = self.config
        track_ids = [track.track_id for track in self.tracks.values() if track.state != "removed"]
        active = [track_id for track_id in track_ids if self.tracks[track_id].state == "tracked"]
        lost = [track_id for track_id in track_ids if self.tracks[track_id].state == "lost"]
        tentative = [track_id for track_id in track_ids if self.tracks[track_id].state == "tentative"]
        high = _score_indices(detections, lambda score: score >= cfg.track_high_thresh)
        low = _score_indices(
            detections,
            lambda score: cfg.track_low_thresh <= score < cfg.track_high_thresh,
        )

        # First associate currently tracked objects using observation direction.
        matches, unmatched_active, unmatched_high = self._assignment(
            active, high, detections, mode="ocsort"
        )
        # Lost trajectories get a separate, more permissive observation recovery.
        lost_matches, unmatched_lost, unmatched_high = self._assignment(
            lost, unmatched_high, detections, mode="ocsort_recovery",
            max_cost=max(cfg.match_thresh, 0.90),
        )
        matches.extend(lost_matches)

        if cfg.use_byte:
            low_matches, unmatched_active, unmatched_low = self._assignment(
                unmatched_active, low, detections, mode="low", max_cost=0.5
            )
            matches.extend(low_matches)
        else:
            unmatched_low = low

        tentative_matches, unmatched_tentative, unmatched_high = self._assignment(
            tentative, unmatched_high, detections, mode="ocsort", max_cost=0.7
        )
        matches.extend(tentative_matches)
        return (
            matches,
            _unique(unmatched_active + unmatched_lost + unmatched_tentative),
            _unique(unmatched_high + unmatched_low),
        )

    def _update_matched_track(self, track: Track, detection: PoseDetection, frame_index: int) -> None:
        track.reupdate_observations(detection, frame_index, self.config)


class DeepOCSORT(OCSORT):
    """Pose-centric OC-SORT with optional external appearance embeddings."""


class FastTracker(ByteTrack):
    """Pose association with occlusion rollback and reappearance handling.

    Occlusion area is measured from an explicit box when available, otherwise
    from a loose extent derived from visible keypoints.
    """

    def _handle_unmatched(
        self,
        unmatched_track_ids,
        unmatched_detection_indices,
        matched_track_ids,
        matched_detection_indices,
        detections,
        frame_index,
    ) -> None:
        del unmatched_detection_indices, matched_detection_indices, detections
        cfg = self.config
        active_boxes = [
            self.tracks[track_id].bbox.copy()
            for track_id in matched_track_ids
            if self.tracks[track_id].state == "tracked"
            and not self.tracks[track_id].is_occluded
        ]
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            if track.state == "removed":
                continue
            if track.was_recently_occluded and frame_index - track.last_occluded_frame > cfg.occ_reappear_window:
                track.was_recently_occluded = False

            covered = any(
                self._coverage(track.bbox, active_box) > cfg.occ_cover_thresh
                for active_box in active_boxes
            )
            if covered:
                if not track.is_occluded:
                    self._rollback_for_occlusion(track, frame_index)
                    track.is_occluded = True
                    track.was_recently_occluded = True
                    track.last_occluded_frame = int(frame_index)
                    track.occluded_len = 1
            elif track.is_occluded:
                track.occluded_len += 1

    @staticmethod
    def _coverage(track_box, detection_box) -> float:
        ax1, ay1, ax2, ay2 = map(float, track_box)
        bx1, by1, bx2, by2 = map(float, detection_box)
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        return intersection / area if area > 0 else 0.0

    def _rollback_for_occlusion(self, track: Track, frame_index: int) -> None:
        history = list(track.state_history)
        if not history:
            return
        pos_target = int(frame_index) - int(self.config.reset_pos_offset_occ)
        vel_target = int(frame_index) - int(self.config.reset_velocity_offset_occ)
        pos_state = next((entry for entry in reversed(history) if entry[0] <= pos_target), history[0])
        vel_state = next((entry for entry in reversed(history) if entry[0] <= vel_target), pos_state)

        # Restore position and motion from independently configurable recent states.
        track.kalman.mean = pos_state[1].copy()
        track.kalman.mean[4:] = vel_state[1][4:]
        track.kalman.covariance = pos_state[2].copy()
        rebuilt = [entry for entry in history if entry[0] <= pos_state[0]]
        for step in range(int(pos_state[0]) + 1, int(frame_index) + 1):
            track.kalman.predict(1)
            rebuilt.append((step, track.kalman.mean.copy(), track.kalman.covariance.copy()))

        track.kalman.mean[4:] *= float(self.config.dampen_motion_occ)
        track.kalman.mean[3] *= float(self.config.enlarge_bbox_occ)
        if rebuilt and rebuilt[-1][0] == int(frame_index):
            rebuilt[-1] = (int(frame_index), track.kalman.mean.copy(), track.kalman.covariance.copy())
        else:
            rebuilt.append((int(frame_index), track.kalman.mean.copy(), track.kalman.covariance.copy()))
        track.state_history = deque(rebuilt, maxlen=track.state_history.maxlen)
        if track.pose_kalman is not None and track.pose_state_history:
            pose_history = list(track.pose_state_history)
            pose_pos = next(
                (entry for entry in reversed(pose_history) if entry[0] <= pos_target),
                pose_history[0],
            )
            pose_vel = next(
                (entry for entry in reversed(pose_history) if entry[0] <= vel_target),
                pose_pos,
            )
            track.pose_kalman.restore(pose_pos[1])
            track.pose_kalman.mean[:, 2:] = pose_vel[1][0][:, 2:]
            pose_rebuilt = [entry for entry in pose_history if entry[0] <= pose_pos[0]]
            for step in range(int(pose_pos[0]) + 1, int(frame_index) + 1):
                track.pose_kalman.predict(1, Track._scale_from_box(track.kalman.bbox))
                pose_rebuilt.append((step, track.pose_kalman.snapshot()))
            track.pose_kalman.mean[:, 2:] *= float(self.config.dampen_motion_occ)
            if pose_rebuilt and pose_rebuilt[-1][0] == int(frame_index):
                pose_rebuilt[-1] = (int(frame_index), track.pose_kalman.snapshot())
            else:
                pose_rebuilt.append((int(frame_index), track.pose_kalman.snapshot()))
            track.pose_state_history = deque(pose_rebuilt, maxlen=track.pose_state_history.maxlen)
        track.last_state_frame = int(frame_index)

    def _should_mark_lost(self, track: Track, frame_index: int) -> bool:
        del frame_index
        if track.state == "tentative":
            return True
        if track.state == "tracked" and track.not_matched <= 2:
            return False
        if track.is_occluded and track.occluded_len <= self.config.active_occ_to_lost_thresh:
            return False
        return True

    def _expire_tracks(self, frame_index: int) -> None:
        for track in self.tracks.values():
            if track.state != "lost":
                continue
            recently_occluded = (
                track.was_recently_occluded
                and frame_index - track.last_occluded_frame <= self.config.occ_reappear_window
            )
            if not recently_occluded and track.time_since_update > self.config.track_buffer:
                track.mark_removed()
            elif track.was_recently_occluded and not recently_occluded:
                track.was_recently_occluded = False

    def _can_start_track(self, detection: PoseDetection, current) -> bool:
        if not super()._can_start_track(detection, current):
            return False
        for track in self.tracks.values():
            if track.state not in {"tracked", "lost"} or track.class_id != detection.class_id:
                continue
            if detection.bbox_xyxy is not None and track.last_detection.bbox_xyxy is not None:
                if bbox_iou(track.bbox, detection.bbox_xyxy) >= self.config.init_iou_suppress:
                    return False
            elif detection.keypoints is not None and track.predicted_keypoints is not None:
                similarity = pose_similarity(
                    track.predicted_keypoints, detection, self.config,
                    first_bbox=track.bbox, second_bbox=detection.bbox_xyxy,
                )
                if np.isfinite(similarity) and similarity >= self.config.init_pose_suppress:
                    return False
        return True


class TrackTrack(BaseTracker):
    """Pose/ReID multi-cue mutual-nearest assignment with pose-based TAI."""

    def _iterative_assignment(self, track_ids, detection_indices, detections, mode, threshold):
        original_tracks = list(track_ids)
        original_detections = list(detection_indices)
        if not original_tracks or not original_detections:
            return [], original_tracks, original_detections
        costs = self._association_cost_matrix(
            original_tracks, original_detections, detections, mode
        )
        remaining_rows = list(range(len(original_tracks)))
        remaining_cols = list(range(len(original_detections)))
        matches = []
        step = float(self.config.reduce_step)
        current_threshold = float(threshold)
        while remaining_rows and remaining_cols and current_threshold >= 0.0:
            local_costs = costs[np.ix_(remaining_rows, remaining_cols)]
            nearest_cols = np.argmin(local_costs, axis=1)
            nearest_rows = np.argmin(local_costs, axis=0)
            paired = [
                (row, int(nearest_cols[row]))
                for row in range(len(remaining_rows))
                if nearest_rows[int(nearest_cols[row])] == row
                and local_costs[row, int(nearest_cols[row])] < current_threshold
            ]
            if not paired:
                break
            accepted_rows, accepted_cols = set(), set()
            for row, col in paired:
                matches.append((
                    original_tracks[remaining_rows[row]],
                    original_detections[remaining_cols[col]],
                ))
                accepted_rows.add(remaining_rows[row])
                accepted_cols.add(remaining_cols[col])
            remaining_rows = [row for row in remaining_rows if row not in accepted_rows]
            remaining_cols = [col for col in remaining_cols if col not in accepted_cols]
            current_threshold -= step
        return (
            matches,
            [original_tracks[row] for row in remaining_rows],
            [original_detections[col] for col in remaining_cols],
        )

    def _associate(self, detections: Sequence[PoseDetection]):
        cfg = self.config
        all_tracks = [track.track_id for track in self.tracks.values() if track.state != "removed"]
        confirmed = [
            track_id for track_id in all_tracks
            if self.tracks[track_id].state in {"tracked", "lost"}
        ]
        tentative = [track_id for track_id in all_tracks if self.tracks[track_id].state == "tentative"]
        high = _score_indices(detections, lambda score: score >= cfg.track_high_thresh)
        low = _score_indices(
            detections,
            lambda score: cfg.track_low_thresh <= score < cfg.track_high_thresh,
        )

        matches, unmatched_confirmed, unmatched_detections = self._iterative_assignment(
            confirmed, high + low, detections, "tracktrack", cfg.match_thresh
        )
        leftover_high = [index for index in unmatched_detections if index in set(high)]
        leftover_low = [index for index in unmatched_detections if index in set(low)]

        tentative_matches, unmatched_tentative, leftover_high = self._iterative_assignment(
            tentative, leftover_high, detections, "tracktrack", cfg.match_thresh
        )
        matches.extend(tentative_matches)
        leftover = _unique(leftover_high + leftover_low)

        if cfg.lost_match_thr > 0.0 and leftover:
            unmatched_lost = [
                track_id for track_id in unmatched_confirmed
                if self.tracks[track_id].state == "lost"
            ]
            lost_matches, _remaining_lost, leftover = self._iterative_assignment(
                unmatched_lost,
                leftover,
                detections,
                "tracktrack_recovered",
                cfg.lost_match_thr,
            )
            matches.extend(lost_matches)
            matched_lost = {track_id for track_id, _index in lost_matches}
            unmatched_confirmed = [
                track_id for track_id in unmatched_confirmed if track_id not in matched_lost
            ]

        return (
            matches,
            _unique(unmatched_confirmed + unmatched_tentative),
            sorted(
                _unique(leftover),
                key=lambda index: detections[index].score,
                reverse=True,
            ),
        )

    def _can_start_track(self, detection: PoseDetection, current) -> bool:
        if not super()._can_start_track(detection, current):
            return False
        active_tracks = [
            track for track in self.tracks.values()
            if track.state == "tracked" and track.class_id == detection.class_id
        ]
        active_tracks.extend(
            track for track in current.values()
            if track.state != "removed" and track.class_id == detection.class_id
        )
        for track in active_tracks:
            similarity = pose_similarity(
                track.predicted_keypoints, detection, self.config,
                first_bbox=track.bbox, second_bbox=detection.bbox_xyxy,
            )
            if np.isfinite(similarity) and similarity > self.config.tai_oks_thr:
                return False
            if (
                not np.isfinite(similarity)
                and detection.bbox_xyxy is not None
                and track.last_detection.bbox_xyxy is not None
                and bbox_iou(track.bbox, detection.bbox_xyxy) > self.config.init_iou_suppress
            ):
                return False
        return True


_TRACKER_TYPES = {
    "bytetrack": ByteTrack,
    "botsort": BoTSORT,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
    "fasttrack": FastTracker,
    "tracktrack": TrackTrack,
}


def create_tracker(config=None, **overrides) -> BaseTracker:
    """Build a standalone tracker from a name, mapping, or ``TrackerConfig``.

    ``create_tracker()`` defaults to ByteTrack. Example::

        tracker = create_tracker({"algorithm": "ocsort", "track_buffer": 45})
        tracked = tracker.update(
            [{"score": 0.9, "keypoints": [
                [20, 30, 0.9], [25, 34, 0.8], [28, 38, 0.9]
            ]}],
            frame_index=0,
        )
    """
    if isinstance(config, TrackerConfig):
        if overrides:
            values = {key: getattr(config, key) for key in config.__dataclass_fields__}
            values.update(overrides)
            config = TrackerConfig.from_mapping(values)
        else:
            config.validate()
    elif config is None:
        config = TrackerConfig.for_algorithm("bytetrack", **overrides)
    elif isinstance(config, str):
        config = TrackerConfig.for_algorithm(config, **overrides)
    else:
        values = dict(config)
        values.update(overrides)
        config = TrackerConfig.from_mapping(values)
    return _TRACKER_TYPES[config.algorithm](config)


__all__ = [
    "ByteTrack",
    "BoTSORT",
    "OCSORT",
    "DeepOCSORT",
    "FastTracker",
    "TrackTrack",
    "create_tracker",
]
