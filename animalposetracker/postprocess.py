"""Compose pose detections with tracking, temporal filtering, and 3D reconstruction.

The functions here form an optional analysis layer after model prediction. They
do not load models, read videos, or modify GUI/live-inference behavior.
"""

from copy import deepcopy
import json
from typing import Any, Dict, Mapping, Optional, Sequence
from pathlib import Path

import numpy as np

from .pose_filters import filter_pose_2d
from .pose3d import CameraCalibration, triangulate_sequence
from .tracking import PoseDetection


def track_frame_predictions(
    detections: Sequence[Mapping[str, Any]],
    tracker,
    frame_index: int,
    frame=None,
    class_names: Optional[Sequence[str]] = None,
) -> list:
    """Attach sequence-local track IDs to one frame of pose predictions."""
    observations = [PoseDetection.from_mapping(item) for item in detections]
    tracked = tracker.update(observations, frame_index=frame_index, frame=frame)
    names = list(class_names or [])
    results = []
    for item in tracked:
        record = item.to_mapping()
        record["bbox_xyxy"] = (
            None if item.bbox_xyxy is None else list(item.bbox_xyxy)
        )
        record["keypoints"] = (
            None if item.keypoints is None
            else np.asarray(item.keypoints, dtype=np.float32).tolist()
        )
        record["embedding"] = (
            None if item.embedding is None
            else np.asarray(item.embedding, dtype=np.float32).tolist()
        )
        record["class_name"] = (
            names[item.class_id] if 0 <= item.class_id < len(names)
            else str(item.class_id)
        )
        results.append(record)
    return results


def filter_tracked_video_records(
    frame_records: Sequence[Mapping[str, Any]],
    method: str = "median",
    window_length: int = 5,
    polyorder: int = 2,
    confidence_threshold: Optional[float] = 0.25,
    max_gap: int = 0,
) -> list:
    """Smooth each track independently while preserving the input record structure.

    Only existing detections are returned. Gap interpolation can inform smoothing
    across a short gap, but this function does not synthesize new detections.
    """
    records = deepcopy(list(frame_records))
    if not records:
        return records
    by_track: Dict[int, Dict[int, Dict[str, Any]]] = {}
    frame_numbers = []
    for frame_record in records:
        if "frame" not in frame_record:
            raise ValueError("video frame records must contain a frame index")
        frame_index = int(frame_record["frame"])
        frame_numbers.append(frame_index)
        for detection in frame_record.get("predictions", []):
            if "track_id" not in detection:
                raise ValueError("temporal filtering requires tracked detections with track_id")
            track_id = int(detection["track_id"])
            if track_id in by_track and frame_index in by_track[track_id]:
                raise ValueError(f"track {track_id} has multiple detections in frame {frame_index}")
            by_track.setdefault(track_id, {})[frame_index] = detection

    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError("video frame records must have unique frame indices")
    frame_count = max(frame_numbers, default=-1) + 1
    for track_id, observations in by_track.items():
        first_record = next(iter(observations.values()))
        first = np.asarray(first_record["keypoints"], dtype=np.float32)
        if first.ndim != 2 or first.shape[1] not in (2, 3):
            raise ValueError(f"track {track_id} keypoints must have shape (K, 2|3)")
        sequence = np.full((frame_count, *first.shape), np.nan, dtype=np.float32)
        for frame_index, detection in observations.items():
            keypoints = np.asarray(detection["keypoints"], dtype=np.float32)
            if keypoints.shape != first.shape:
                raise ValueError(f"track {track_id} changes keypoint shape within the video")
            sequence[frame_index] = keypoints

        threshold = confidence_threshold if first.shape[1] == 3 else None
        filtered = filter_pose_2d(
            sequence,
            method=method,
            window_length=window_length,
            polyorder=polyorder,
            confidence_threshold=threshold,
            max_gap=max_gap,
        )
        for frame_index, detection in observations.items():
            updated = np.asarray(detection["keypoints"], dtype=np.float32).copy()
            coordinates = filtered[frame_index, :, :2]
            valid = np.isfinite(coordinates).all(axis=1)
            updated[valid, :2] = coordinates[valid]
            detection["keypoints"] = updated.tolist()
    return records


def triangulate_multiview_predictions(
    predictions_by_camera: Mapping[str, Any],
    cameras: Mapping[str, Any],
    identity_map: Optional[Mapping[str, Mapping[Any, str]]] = None,
    source_by_camera: Optional[Mapping[str, str]] = None,
    **triangulation_options,
) -> Dict[str, Dict[str, Any]]:
    """Triangulate synchronized tracked video records across camera views.

    Each value in ``predictions_by_camera`` is the frame-record list from a
    prediction result. Single-animal sequences are matched automatically. For
    multi-animal videos, pass ``identity_map[camera_id][local_track_id]`` to map
    camera-local track IDs to the same cross-camera animal identity.
    """
    if len(predictions_by_camera) < 2:
        raise ValueError("3D triangulation requires predictions from at least two cameras")
    camera_models = {
        str(camera_id): (
            value if isinstance(value, CameraCalibration)
            else CameraCalibration.from_mapping(value)
        )
        for camera_id, value in cameras.items()
    }
    if set(camera_models) != {str(camera_id) for camera_id in predictions_by_camera}:
        raise ValueError("camera calibration IDs must exactly match prediction camera IDs")

    frames_by_camera: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    local_ids_by_camera: Dict[str, set] = {}
    for camera_id, entries in predictions_by_camera.items():
        camera_id = str(camera_id)
        if isinstance(entries, (str, Path)):
            with Path(entries).expanduser().open("r", encoding="utf-8") as stream:
                entries = json.load(stream)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise TypeError(f"camera {camera_id} predictions must be a frame-record sequence or JSON path")
        frame_records = {}
        local_ids = set()
        requested_source = (source_by_camera or {}).get(camera_id)
        if requested_source is not None:
            requested_source = str(Path(requested_source).expanduser().resolve())
        for record in entries:
            if "frame" not in record:
                continue
            if requested_source is not None:
                record_source = record.get("source")
                if record_source is None or str(Path(record_source).expanduser().resolve()) != requested_source:
                    continue
            frame_index = int(record["frame"])
            if frame_index in frame_records:
                raise ValueError(f"camera {camera_id} has duplicate frame {frame_index}")
            frame_records[frame_index] = record
            detections = record.get("predictions", [])
            for detection in detections:
                if "track_id" in detection:
                    local_ids.add(str(detection["track_id"]))
                elif len(detections) == 1:
                    local_ids.add("__single_animal__")
                else:
                    raise ValueError(
                        f"camera {camera_id} has multiple untracked poses in frame {frame_index}; "
                        "track them before triangulation"
                    )
        if not frame_records:
            suffix = f" for source {requested_source}" if requested_source else ""
            raise ValueError(f"camera {camera_id} has no frame prediction records{suffix}")
        frames_by_camera[camera_id] = frame_records
        local_ids_by_camera[camera_id] = local_ids

    frame_sets = [set(records) for records in frames_by_camera.values()]
    if any(frame_set != frame_sets[0] for frame_set in frame_sets[1:]):
        raise ValueError("camera videos must have matching synchronized frame indices")
    frame_indices = sorted(frame_sets[0])

    identities_by_camera: Dict[str, Dict[str, str]] = {}
    for camera_id, local_ids in local_ids_by_camera.items():
        mapping = (identity_map or {}).get(camera_id, {})
        if mapping:
            normalized = {str(local_id): str(identity) for local_id, identity in mapping.items()}
            missing = sorted(local_ids - set(normalized))
            if missing:
                raise ValueError(
                    f"camera {camera_id} identity_map is missing local track IDs: {missing}"
                )
            identities_by_camera[camera_id] = normalized
        elif len(local_ids) <= 1:
            identities_by_camera[camera_id] = {
                local_id: "animal_0" for local_id in local_ids
            }
        else:
            raise ValueError(
                f"camera {camera_id} has multiple local tracks; provide identity_map "
                "to match animals across cameras"
            )

    global_identities = sorted({
        identity
        for mapping in identities_by_camera.values()
        for identity in mapping.values()
    })
    results: Dict[str, Dict[str, Any]] = {}
    for identity in global_identities:
        sequence_by_camera = {}
        keypoint_shape = None
        for camera_id, frame_records in frames_by_camera.items():
            local_to_global = identities_by_camera[camera_id]
            for frame_record in frame_records.values():
                for detection in frame_record.get("predictions", []):
                    local_id = str(detection.get("track_id", "__single_animal__"))
                    if local_to_global.get(local_id) != identity:
                        continue
                    keypoints = np.asarray(detection["keypoints"], dtype=np.float32)
                    if keypoint_shape is None:
                        keypoint_shape = keypoints.shape
                    elif keypoints.shape != keypoint_shape:
                        raise ValueError("keypoint shapes differ across cameras or frames")

        if keypoint_shape is None:
            continue
        for camera_id, frame_records in frames_by_camera.items():
            local_to_global = identities_by_camera[camera_id]
            sequence = np.full((len(frame_indices), *keypoint_shape), np.nan, dtype=np.float32)
            for position, frame_index in enumerate(frame_indices):
                frame_record = frame_records[frame_index]
                matches = []
                for detection in frame_record.get("predictions", []):
                    local_id = str(detection.get("track_id", "__single_animal__"))
                    if local_to_global.get(local_id) == identity:
                        matches.append(detection)
                if len(matches) > 1:
                    raise ValueError(
                        f"camera {camera_id} has multiple detections for {identity} at frame {frame_index}"
                    )
                if matches:
                    sequence[position] = np.asarray(matches[0]["keypoints"], dtype=np.float32)
            sequence_by_camera[camera_id] = sequence

        results[identity] = {
            "frame_indices": frame_indices,
            "sequence": triangulate_sequence(
                sequence_by_camera, camera_models, **triangulation_options
            ),
        }
    return results


__all__ = [
    "track_frame_predictions",
    "filter_tracked_video_records",
    "triangulate_multiview_predictions",
]
