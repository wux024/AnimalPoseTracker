"""State and lifecycle management shared by the standalone tracker algorithms."""

from collections import deque
from typing import Sequence

import numpy as np

from .camera_motion import CameraMotionEstimator
from .config import TrackerConfig
from .geometry import (
    bbox_iou,
    keypoints_to_bbox,
    pose_similarity,
)
from .kalman import KalmanBoxFilter, KalmanKeypointFilter
from .types import PoseDetection, TrackedDetection


def _warp_box(box, matrix):
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float32)
    corners = np.asarray([
        [box[0], box[1], 1.0], [box[2], box[1], 1.0],
        [box[2], box[3], 1.0], [box[0], box[3], 1.0],
    ], dtype=np.float32)
    transformed = corners @ np.asarray(matrix, dtype=np.float32).T
    return (
        float(transformed[:, 0].min()), float(transformed[:, 1].min()),
        float(transformed[:, 0].max()), float(transformed[:, 1].max()),
    )


def _warp_points(points, matrix):
    if points is None:
        return None
    warped = np.asarray(points, dtype=np.float32).copy()
    homogeneous = np.concatenate(
        [warped[:, :2], np.ones((len(warped), 1), dtype=np.float32)], axis=1
    )
    warped[:, :2] = homogeneous @ np.asarray(matrix, dtype=np.float32).T
    return warped


def _interpolate_keypoints(first, second, fraction):
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.shape[1] == 2:
        left = np.column_stack([left, np.ones((len(left),), dtype=np.float32)])
    if right.shape[1] == 2:
        right = np.column_stack([right, np.ones((len(right),), dtype=np.float32)])
    common = (left[:, 2] > 0.0) & (right[:, 2] > 0.0)
    result = np.zeros_like(left)
    result[common, :2] = (1.0 - fraction) * left[common, :2] + fraction * right[common, :2]
    result[common, 2] = np.minimum(left[common, 2], right[common, 2])
    return result


def _effective_box(detection: PoseDetection, confidence_threshold: float = 0.0):
    if detection.bbox_xyxy is not None:
        return np.asarray(detection.bbox_xyxy, dtype=np.float32)
    if detection.keypoints is None:
        raise ValueError("a detection without bbox_xyxy must include keypoints")
    return keypoints_to_bbox(detection.keypoints, 0.0)


def _warp_detection(detection: PoseDetection, matrix: np.ndarray) -> PoseDetection:
    warped_box = _warp_box(detection.bbox_xyxy, matrix)
    keypoints = detection.keypoints
    if keypoints is not None:
        keypoints = _warp_points(keypoints, matrix)
    return PoseDetection(
        bbox_xyxy=warped_box,
        score=detection.score,
        class_id=detection.class_id,
        keypoints=keypoints,
        embedding=detection.embedding,
    )


class Track:
    """One internal trajectory with a Kalman box state and bounded observations."""

    def __init__(self, track_id: int, detection: PoseDetection, frame_index: int, config: TrackerConfig):
        self.track_id = int(track_id)
        self.kalman = KalmanBoxFilter(_effective_box(detection, config.pose_keypoint_thresh))
        self.pose_kalman = (
            KalmanKeypointFilter(
                detection.keypoints,
                confidence_threshold=config.pose_keypoint_thresh,
                scale=self._scale_from_box(_effective_box(detection, config.pose_keypoint_thresh)),
            )
            if detection.keypoints is not None else None
        )
        self.class_id = int(detection.class_id)
        self.score = float(detection.score)
        self.previous_score = float(detection.score)
        self.last_detection = detection
        self.embedding = None if detection.embedding is None else detection.embedding.copy()
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.start_frame = int(frame_index)
        self.last_frame = int(frame_index)
        self.last_state_frame = int(frame_index)
        self.state = "tracked" if config.min_track_len <= 1 else "tentative"
        history_len = max(
            8,
            int(config.track_buffer) + 2,
            int(config.reset_velocity_offset_occ) + 4,
            int(config.reset_pos_offset_occ) + 4,
            int(config.delta_t) + 3,
        )
        self.state_history = deque(maxlen=history_len)
        self.observation_history = deque(maxlen=max(int(config.delta_t) + 3, 6))
        self.state_history.append((int(frame_index), self.kalman.mean.copy(), self.kalman.covariance.copy()))
        self.observation_history.append((int(frame_index), _effective_box(detection, config.pose_keypoint_thresh)))
        self.pose_observation_history = deque(maxlen=max(int(config.delta_t) + 3, 6))
        if detection.keypoints is not None:
            self.pose_observation_history.append((int(frame_index), detection.keypoints.copy()))
        self.pose_state_history = deque(maxlen=history_len)
        if self.pose_kalman is not None:
            self.pose_state_history.append((int(frame_index), self.pose_kalman.snapshot()))

        # FastTracker bookkeeping.
        self.is_occluded = False
        self.occluded_len = 0
        self.was_recently_occluded = False
        self.last_occluded_frame = -1
        self.not_matched = 0

    @staticmethod
    def _scale_from_box(box):
        box = np.asarray(box, dtype=np.float32)
        return max(float(np.sqrt(max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0))), 1.0)

    @property
    def bbox(self) -> np.ndarray:
        return self.kalman.bbox

    @property
    def predicted_keypoints(self):
        return None if self.pose_kalman is None else self.pose_kalman.as_keypoints()

    @property
    def confirmed(self) -> bool:
        return self.state == "tracked" and self.hits > 0

    def predict(self, frame_index: int) -> None:
        frame_index = int(frame_index)
        delta = frame_index - self.last_state_frame
        if delta < 1:
            raise ValueError("frame_index must increase between tracker updates")
        for step in range(1, delta + 1):
            self.kalman.predict(1)
            if self.pose_kalman is not None:
                self.pose_kalman.predict(1, self._scale_from_box(self.kalman.bbox))
            self.age += 1
            state_frame = self.last_state_frame + step
            self.state_history.append((state_frame, self.kalman.mean.copy(), self.kalman.covariance.copy()))
            if self.pose_kalman is not None:
                self.pose_state_history.append((state_frame, self.pose_kalman.snapshot()))
        self.time_since_update += delta
        self.not_matched += delta
        self.last_state_frame = frame_index

    def update(self, detection: PoseDetection, frame_index: int, config: TrackerConfig) -> None:
        frame_index = int(frame_index)
        effective_box = _effective_box(detection, config.pose_keypoint_thresh)
        self.kalman.update(effective_box)
        if detection.keypoints is not None:
            if self.pose_kalman is None:
                self.pose_kalman = KalmanKeypointFilter(
                    detection.keypoints,
                    confidence_threshold=config.pose_keypoint_thresh,
                    scale=self._scale_from_box(effective_box),
                )
            else:
                self.pose_kalman.update(
                    detection.keypoints,
                    confidence_threshold=config.pose_keypoint_thresh,
                    scale=self._scale_from_box(effective_box),
                )
            self.pose_observation_history.append((frame_index, detection.keypoints.copy()))
        self.class_id = int(detection.class_id)
        self.previous_score = self.score
        self.score = float(detection.score)
        self.last_detection = detection
        self.last_frame = frame_index
        self.last_state_frame = frame_index
        self.time_since_update = 0
        self.not_matched = 0
        self.hits += 1
        self.state = "tracked" if self.hits >= config.min_track_len else "tentative"
        self.observation_history.append((frame_index, np.asarray(effective_box, dtype=np.float32)))
        self.state_history.append((frame_index, self.kalman.mean.copy(), self.kalman.covariance.copy()))
        if self.pose_kalman is not None:
            self.pose_state_history.append((frame_index, self.pose_kalman.snapshot()))
        if detection.embedding is not None:
            if self.embedding is None:
                self.embedding = detection.embedding.copy()
            else:
                alpha = float(config.alpha_fixed_emb)
                updated = alpha * self.embedding + (1.0 - alpha) * detection.embedding
                norm = float(np.linalg.norm(updated))
                self.embedding = updated / norm if norm > 0.0 else updated
        self.is_occluded = False
        self.occluded_len = 0
        self.was_recently_occluded = False
        self.last_occluded_frame = -1

    def mark_lost(self) -> None:
        if self.state != "removed":
            self.state = "lost"

    def mark_removed(self) -> None:
        self.state = "removed"

    def apply_affine(self, matrix: np.ndarray) -> None:
        self.kalman.apply_affine(matrix)
        if self.pose_kalman is not None:
            self.pose_kalman.apply_affine(matrix)
        self.last_detection = _warp_detection(self.last_detection, matrix)
        self.observation_history = deque(
            ((frame, _warp_box(box, matrix)) for frame, box in self.observation_history),
            maxlen=self.observation_history.maxlen,
        )
        self.pose_observation_history = deque(
            ((frame, _warp_points(points, matrix)) for frame, points in self.pose_observation_history),
            maxlen=self.pose_observation_history.maxlen,
        )
        self.state_history.clear()
        self.state_history.append((
            self.last_state_frame,
            self.kalman.mean.copy(),
            self.kalman.covariance.copy(),
        ))
        if self.pose_kalman is not None:
            self.pose_state_history.clear()
            self.pose_state_history.append((self.last_state_frame, self.pose_kalman.snapshot()))

    def reupdate_observations(self, detection: PoseDetection, frame_index: int, config: TrackerConfig) -> None:
        """Rebuild box and pose motion states with virtual observations across a lost interval."""
        if not self.observation_history:
            self.update(detection, frame_index, config)
            return
        last_frame, last_box = self.observation_history[-1]
        gap = int(frame_index) - int(last_frame)
        if gap <= 1:
            self.update(detection, frame_index, config)
            return
        history_state = next(
            ((mean.copy(), covariance.copy()) for frame, mean, covariance in reversed(self.state_history)
             if frame == last_frame),
            None,
        )
        if history_state is not None:
            self.kalman.mean, self.kalman.covariance = history_state
        pose_frame = None
        pose_points = None
        if self.pose_kalman is not None and self.pose_observation_history:
            pose_frame, pose_points = self.pose_observation_history[-1]
            pose_state = next(
                (snapshot for frame, snapshot in reversed(self.pose_state_history) if frame == pose_frame),
                None,
            )
            if pose_state is not None:
                self.pose_kalman.restore(pose_state)
        last_box = np.asarray(last_box, dtype=np.float32)
        new_box = _effective_box(detection, config.pose_keypoint_thresh)
        for step in range(1, gap):
            ratio = step / float(gap)
            virtual_box = (1.0 - ratio) * last_box + ratio * new_box
            self.kalman.predict(1)
            self.kalman.update(virtual_box)
            if self.pose_kalman is not None:
                self.pose_kalman.predict(1, self._scale_from_box(virtual_box))
                if pose_points is not None and detection.keypoints is not None:
                    virtual_pose = _interpolate_keypoints(pose_points, detection.keypoints, ratio)
                    self.pose_kalman.update(virtual_pose, config.pose_keypoint_thresh, self._scale_from_box(virtual_box))
        self.kalman.predict(1)
        if self.pose_kalman is not None:
            self.pose_kalman.predict(1, self._scale_from_box(new_box))
        self.update(detection, frame_index, config)


class BaseTracker:
    """Online pose-track lifecycle shared by the independent association algorithms."""

    def __init__(self, config: TrackerConfig):
        self.config = config.validate()
        self.tracks = {}
        self.next_track_id = 1
        self.last_frame_index = None
        self.num_keypoints = None
        self.motion = CameraMotionEstimator(self.config.gmc_method)

    def reset(self) -> None:
        """Clear all per-sequence state and restart IDs from one."""
        self.tracks.clear()
        self.next_track_id = 1
        self.last_frame_index = None
        self.num_keypoints = None
        self.motion.reset()

    def update(self, detections: Sequence[PoseDetection], frame_index: int, frame=None):
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.last_frame_index is not None and frame_index <= self.last_frame_index:
            raise ValueError("frame_index must strictly increase; call reset() for a new sequence")
        observations = [
            item if isinstance(item, PoseDetection) else PoseDetection.from_mapping(item)
            for item in detections
        ]
        observed_counts = {item.keypoints.shape[0] for item in observations if item.keypoints is not None}
        if len(observed_counts) > 1:
            raise ValueError("all keypoint detections in one frame must use the same keypoint count")
        if observed_counts:
            count = next(iter(observed_counts))
            if self.num_keypoints is not None and count != self.num_keypoints:
                raise ValueError("keypoint count must stay constant within a tracker sequence")
            if self.config.kpt_oks_sigmas is not None and len(self.config.kpt_oks_sigmas) != count:
                raise ValueError(
                    f"kpt_oks_sigmas has {len(self.config.kpt_oks_sigmas)} values but detections have {count} keypoints"
                )
            self.num_keypoints = count
        if self.config.with_reid and any(item.embedding is None for item in observations):
            raise ValueError("with_reid=True requires an embedding on every input detection")

        if self.config.gmc_method != "none":
            if frame is None:
                raise ValueError("This tracker configuration enables GMC and requires frame=...")
            transform = self.motion.estimate(frame)
            for track in self.tracks.values():
                if track.state != "removed":
                    track.apply_affine(transform)

        active = [track for track in self.tracks.values() if track.state != "removed"]
        for track in active:
            track.predict(frame_index)
        self.last_frame_index = frame_index

        matches, unmatched_track_ids, unmatched_detection_indices = self._associate(observations)
        current = {}
        for track_id, detection_index in matches:
            track = self.tracks[track_id]
            detection = observations[detection_index]
            self._update_matched_track(track, detection, frame_index)
            current[detection_index] = track

        matched_track_ids = {track_id for track_id, _index in matches}
        matched_detection_indices = {index for _track_id, index in matches}
        self._handle_unmatched(
            unmatched_track_ids,
            unmatched_detection_indices,
            matched_track_ids,
            matched_detection_indices,
            observations,
            frame_index,
        )
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            if track.time_since_update > 0:
                if self._should_mark_lost(track, frame_index):
                    if track.state == "tentative":
                        track.mark_removed()
                    else:
                        track.mark_lost()

        for detection_index in unmatched_detection_indices:
            detection = observations[detection_index]
            if self._can_start_track(detection, current):
                track = Track(self.next_track_id, detection, frame_index, self.config)
                self.tracks[track.track_id] = track
                current[detection_index] = track
                matched_track_ids.add(track.track_id)
                self.next_track_id += 1

        self._expire_tracks(frame_index)
        results = []
        for detection_index in sorted(current):
            track = current[detection_index]
            if track.state == "tracked" and track.hits >= self.config.min_track_len:
                results.append(TrackedDetection(
                    detection=observations[detection_index],
                    track_id=track.track_id,
                    age=track.age,
                    hits=track.hits,
                    is_confirmed=True,
                ))
        return results

    def _update_matched_track(self, track: Track, detection: PoseDetection, frame_index: int) -> None:
        track.update(detection, frame_index, self.config)

    def _can_start_track(self, detection: PoseDetection, current) -> bool:
        return detection.score >= max(
            self.config.new_track_thresh,
            self.config.track_high_thresh,
        )

    def _handle_unmatched(
        self,
        unmatched_track_ids,
        unmatched_detection_indices,
        matched_track_ids,
        matched_detection_indices,
        detections,
        frame_index,
    ) -> None:
        del unmatched_track_ids, unmatched_detection_indices, matched_track_ids
        del matched_detection_indices, detections, frame_index

    def _should_mark_lost(self, track: Track, frame_index: int) -> bool:
        del frame_index
        return True

    def _expire_tracks(self, frame_index: int) -> None:
        for track in self.tracks.values():
            if track.state == "lost" and track.time_since_update > self.config.track_buffer:
                track.mark_removed()

    def _associate(self, detections: Sequence[PoseDetection]):
        raise NotImplementedError

    def _assignment(self, track_ids, detection_indices, detections, mode="pose", max_cost=None):
        track_ids = list(track_ids)
        detection_indices = list(detection_indices)
        if not track_ids or not detection_indices:
            return [], track_ids, detection_indices
        costs = self._association_cost_matrix(track_ids, detection_indices, detections, mode)
        threshold = float(self.config.match_thresh if max_cost is None else max_cost)
        local_matches, unmatched_rows, unmatched_cols = _linear_assignment(costs, threshold)
        matches = [
            (track_ids[row], detection_indices[col]) for row, col in local_matches
        ]
        return (
            matches,
            [track_ids[row] for row in unmatched_rows],
            [detection_indices[col] for col in unmatched_cols],
        )

    def _association_cost_matrix(self, track_ids, detection_indices, detections, mode="pose"):
        track_ids = list(track_ids)
        detection_indices = list(detection_indices)
        tracks = [self.tracks[track_id] for track_id in track_ids]
        selected = [detections[index] for index in detection_indices]
        overlaps = np.zeros((len(tracks), len(selected)), dtype=np.float32)
        box_supported = np.zeros_like(overlaps, dtype=bool)
        box_fallback = np.ones_like(overlaps, dtype=bool)
        pose_similarities = np.full_like(overlaps, np.nan, dtype=np.float32)
        for row, track in enumerate(tracks):
            comparison_pose = (
                track.last_detection.keypoints
                if mode == "ocsort_recovery" else track.predicted_keypoints
            )
            if comparison_pose is None:
                comparison_pose = track.last_detection.keypoints
            comparison_box = (
                _effective_box(track.last_detection, self.config.pose_keypoint_thresh)
                if mode == "ocsort_recovery" else track.bbox
            )
            for col, detection in enumerate(selected):
                detection_box = _effective_box(detection, self.config.pose_keypoint_thresh)
                overlaps[row, col] = _box_iou(comparison_box, detection_box)
                box_supported[row, col] = (
                    track.last_detection.bbox_xyxy is not None
                    and detection.bbox_xyxy is not None
                )
                similarity = pose_similarity(
                    comparison_pose,
                    detection,
                    self.config,
                    first_bbox=comparison_box,
                    second_bbox=detection_box,
                )
                if np.isfinite(similarity):
                    pose_similarities[row, col] = similarity

        pose_costs = 1.0 - pose_similarities

        if mode.startswith("tracktrack"):
            costs = _tracktrack_cost(
                tracks, selected, pose_costs, overlaps, box_supported, box_fallback, self.config
            )
            if mode == "tracktrack":
                low_penalty = np.asarray([
                    self.config.penalty_p
                    if detection.score < self.config.track_high_thresh else 0.0
                    for detection in selected
                ], dtype=np.float32)[None, :]
                costs = costs + low_penalty
            elif mode == "tracktrack_recovered":
                low_penalty = np.asarray([
                    self.config.penalty_q
                    if detection.score < self.config.track_high_thresh else 0.0
                    for detection in selected
                ], dtype=np.float32)[None, :]
                costs = costs + low_penalty
        else:
            costs = np.full_like(overlaps, np.inf, dtype=np.float32)
            for row, track in enumerate(tracks):
                for col, detection in enumerate(selected):
                    pose_cost = pose_costs[row, col]
                    has_pose = np.isfinite(pose_cost)
                    has_box = bool(box_supported[row, col])
                    if has_pose:
                        pose_weight = float(self.config.pose_weight)
                        box_weight = float(self.config.box_weight) if has_box else 0.0
                        total_weight = pose_weight + box_weight
                        if total_weight <= 0.0:
                            value = float(pose_cost)
                        else:
                            value = (
                                pose_weight * float(pose_cost)
                                + box_weight * (1.0 - float(overlaps[row, col]))
                            ) / total_weight
                        if self.config.fuse_score and mode != "low":
                            similarity = 1.0 - value
                            value = 1.0 - similarity * float(detection.score)
                        costs[row, col] = value
                    elif box_fallback[row, col]:
                        value = 1.0 - float(overlaps[row, col])
                        if self.config.fuse_score and mode != "low":
                            value = 1.0 - float(overlaps[row, col]) * float(detection.score)
                        costs[row, col] = value
            if mode in {"ocsort", "ocsort_recovery"}:
                costs = _add_pose_direction_cost(
                    costs, tracks, selected, self.config
                )
            if self.config.with_reid:
                proximity = np.where(
                    np.isfinite(pose_similarities), pose_similarities, overlaps
                )
                costs = _add_reid_cost(costs, tracks, selected, proximity, self.config)
        for row, track in enumerate(tracks):
            for col, detection in enumerate(selected):
                if track.class_id != detection.class_id:
                    costs[row, col] = np.inf
        return costs


def _box_iou(first, second) -> float:
    from .geometry import bbox_iou

    return bbox_iou(first, second)


def _linear_assignment(costs, threshold):
    from .geometry import linear_assignment

    return linear_assignment(costs, threshold)


def _add_reid_cost(costs, tracks, detections, overlaps, config):
    result = np.asarray(costs, dtype=np.float32).copy()
    weight = float(config.reid_weight)
    for row, track in enumerate(tracks):
        if track.embedding is None:
            continue
        for col, detection in enumerate(detections):
            proximity = float(overlaps[row, col])
            if (
                detection.embedding is None
                or not np.isfinite(proximity)
                or proximity < config.proximity_thresh
            ):
                continue
            left = track.embedding
            right = detection.embedding
            if left.shape != right.shape:
                raise ValueError("ReID embeddings in a matched pair must have the same dimension")
            similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
            if similarity < config.appearance_thresh:
                result[row, col] = np.inf
                continue
            appearance_cost = (1.0 - similarity) * 0.5
            result[row, col] = (1.0 - weight) * result[row, col] + weight * appearance_cost
    return result


def _pose_direction_cost(track: Track, detection: PoseDetection, config: TrackerConfig):
    history = list(track.pose_observation_history)
    if len(history) < 2 or detection.keypoints is None:
        return float("nan")
    last_frame, last_points = history[-1]
    previous_frame = int(last_frame) - int(config.delta_t)
    previous_entry = next(
        (entry for entry in reversed(history[:-1]) if int(entry[0]) <= previous_frame),
        history[0],
    )
    previous_points = previous_entry[1]
    current_points = detection.keypoints
    if previous_points.shape[0] != last_points.shape[0] or current_points.shape[0] != last_points.shape[0]:
        return float("nan")

    def confidence(points):
        return points[:, 2] if points.shape[1] == 3 else np.ones((len(points),), dtype=np.float32)

    previous_conf = confidence(previous_points)
    last_conf = confidence(last_points)
    current_conf = confidence(current_points)
    visible = (
        (previous_conf >= config.pose_keypoint_thresh)
        & (last_conf >= config.pose_keypoint_thresh)
        & (current_conf >= config.pose_keypoint_thresh)
    )
    if not visible.any():
        return float("nan")
    track_motion = last_points[visible, :2] - previous_points[visible, :2]
    detection_motion = current_points[visible, :2] - last_points[visible, :2]
    track_norm = np.linalg.norm(track_motion, axis=1)
    detection_norm = np.linalg.norm(detection_motion, axis=1)
    moving = (track_norm > 1e-5) & (detection_norm > 1e-5)
    if not moving.any():
        return float("nan")
    cosine = np.sum(
        track_motion[moving] * detection_motion[moving], axis=1
    ) / (track_norm[moving] * detection_norm[moving])
    cost = float(np.mean((1.0 - np.clip(cosine, -1.0, 1.0)) * 0.5))
    return float(np.clip(cost * detection.score, 0.0, 1.0))


def _add_pose_direction_cost(costs, tracks, detections, config):
    result = np.asarray(costs, dtype=np.float32).copy()
    for row, track in enumerate(tracks):
        for col, detection in enumerate(detections):
            direction_cost = _pose_direction_cost(track, detection, config)
            if np.isfinite(direction_cost) and np.isfinite(result[row, col]):
                result[row, col] += float(config.inertia) * direction_cost
    return result


def _tracktrack_cost(tracks, detections, pose_costs, overlaps, box_supported, box_fallback, config):
    """Pose-first multi-cue association with optional bbox/ReID terms."""
    costs = np.full_like(pose_costs, np.inf, dtype=np.float32)
    for row, track in enumerate(tracks):
        for col, detection in enumerate(detections):
            component_costs = []
            component_weights = []
            pose_valid = np.isfinite(pose_costs[row, col])
            if pose_valid and config.pose_weight > 0.0:
                component_costs.append(float(pose_costs[row, col]))
                component_weights.append(float(config.pose_weight))
            if box_supported[row, col] and config.box_weight > 0.0:
                component_costs.append(1.0 - float(overlaps[row, col]))
                component_weights.append(float(config.box_weight))
            if not component_costs and box_fallback[row, col]:
                # Preserve box-only detections as a fallback; pose remains the
                # primary cost whenever both poses are available.
                component_costs.append(1.0 - float(overlaps[row, col]))
                component_weights.append(1.0)
            elif not component_costs and pose_valid:
                component_costs.append(float(pose_costs[row, col]))
                component_weights.append(1.0)
            if not component_costs:
                continue

            motion_cost = float(np.average(component_costs, weights=component_weights))
            if (
                config.with_reid
                and track.embedding is not None
                and detection.embedding is not None
            ):
                proximity = (
                    float(overlaps[row, col])
                    if box_supported[row, col]
                    else (1.0 - float(pose_costs[row, col]) if np.isfinite(pose_costs[row, col]) else 0.0)
                )
                if proximity >= config.proximity_thresh:
                    if track.embedding.shape != detection.embedding.shape:
                        raise ValueError("ReID embeddings in a matched pair must have the same dimension")
                    similarity = float(np.clip(np.dot(track.embedding, detection.embedding), -1.0, 1.0))
                    if similarity < config.appearance_thresh:
                        costs[row, col] = np.inf
                        continue
                    appearance_cost = (1.0 - similarity) * 0.5
                    reid_weight = float(config.reid_weight)
                    motion_cost = (motion_cost + reid_weight * appearance_cost) / (1.0 + reid_weight)

            projected_score = track.score + (track.score - track.previous_score)
            confidence_cost = abs(projected_score - detection.score)
            angle_cost = _pose_direction_cost(track, detection, config)
            cost = motion_cost + config.conf_weight * confidence_cost
            if np.isfinite(angle_cost):
                cost += config.angle_weight * angle_cost
            costs[row, col] = float(np.clip(cost, 0.0, 1.0))
    return costs


__all__ = ["Track", "BaseTracker"]
