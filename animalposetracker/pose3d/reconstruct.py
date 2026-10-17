"""Multi-view reconstruction for a project: 2D predictions in, ``results/pose3d`` out.

The runner owns the whole 3D stage: it reads one ``predictions.json`` per
camera, associates instances across views (``index`` only for now), undistorts
the pixels with the rig's distortion model, triangulates, and writes the files
agreed for the 3D protocol: ``poses3d.json``, ``poses3d-filter.json`` (only when
filtering is enabled) and ``report.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .calibration import CameraRig, load_camera_rig
from .core import triangulate_keypoints

SCHEMA = "pose3d/1"
ASSOCIATIONS = {"index"}


@dataclass
class _View:
    camera: str
    path: Path
    records: List[Mapping[str, Any]]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _prediction_records(path: Path) -> List[Mapping[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, Mapping):
        payload = payload.get("predictions") or payload.get("frames") or []
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"{path} does not contain a list of prediction records")
    records = list(payload)
    for record in records:
        if isinstance(record, Mapping) and "predictions" not in record and "frames" in record:
            raise ValueError(
                f"{path} holds video predictions; the 3D stage currently needs one record per frame"
            )
    return records


def _instances(record: Mapping[str, Any]) -> List[np.ndarray]:
    """Per-instance ``(K, 3)`` arrays of ``x, y, confidence`` in pixels."""
    poses = []
    for instance in record.get("predictions") or record.get("instances") or []:
        keypoints = instance.get("keypoints")
        if keypoints is None:
            continue
        array = np.asarray(keypoints, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Each prediction's 'keypoints' must be a (K, 2|3) array")
        if array.shape[1] == 2:
            array = np.concatenate([array, np.ones((array.shape[0], 1))], axis=1)
        poses.append(array[:, :3])
    return poses


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def reconstruct_project(
    project_dir: Path | str,
    *,
    calibration_path: Optional[Path | str] = None,
    views: Optional[Mapping[str, Path | str]] = None,
    association: str = "index",
    output_dir: Optional[Path | str] = None,
    confidence_threshold: float = 0.05,
    reprojection_threshold: float = 8.0,
    min_views: int = 2,
    pose_filter: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Triangulate every frame of a 3D-enabled project and write the protocol files."""
    project = Path(project_dir).expanduser().resolve()
    identity = _read_yaml(project / "project.yaml") if (project / "project.yaml").is_file() else {}
    if not identity:
        raise FileNotFoundError(f"Not a project directory (no project.yaml): {project}")
    if not identity.get("pose3d"):
        raise ValueError(
            f"Project '{project.name}' was created without the 3D option "
            "(project.yaml has no pose3d: true); recreate it with --pose3d"
        )
    if association not in ASSOCIATIONS:
        raise ValueError(f"association must be one of {sorted(ASSOCIATIONS)}, got '{association}'")

    rig_path = Path(calibration_path or project / "configs" / "cameras.yaml")
    rig: CameraRig = load_camera_rig(rig_path)

    view_paths: Dict[str, Path] = {}
    for camera in rig.cameras:
        candidate = Path((views or {}).get(camera) or project / "results" / "predict" / camera / "predictions.json")
        if not candidate.is_absolute():
            candidate = project / candidate
        view_paths[camera] = candidate
    missing = [f"{c} -> {p}" for c, p in view_paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing 2D predictions for: " + ", ".join(missing))

    loaded = [_View(camera, view_paths[camera], _prediction_records(view_paths[camera])) for camera in rig.cameras]
    frame_counts = {view.camera: len(view.records) for view in loaded}
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"Views disagree on frame count: {frame_counts}")

    frames: List[Dict[str, Any]] = []
    frames_per_camera = {camera: 0 for camera in rig.cameras}
    total_points = 0
    dropped_points = 0
    errors: List[float] = []

    for frame_index in range(next(iter(frame_counts.values()))):
        per_view = [(view.camera, _instances(view.records[frame_index])) for view in loaded]
        instance_count = max(len(instances) for _, instances in per_view) if per_view else 0
        frame_instances: List[Dict[str, Any]] = []
        for instance_index in range(instance_count):
            poses: Dict[str, np.ndarray] = {}
            for camera, instances in per_view:
                if instance_index < len(instances):
                    poses[camera] = rig.undistort(camera, instances[instance_index])
            if len(poses) < min_views:
                continue
            result = triangulate_keypoints(
                poses, rig.cameras,
                confidence_threshold=confidence_threshold,
                reprojection_threshold=reprojection_threshold,
                min_views=min_views,
            )
            finite = np.isfinite(result.points_3d).all(axis=1)
            total_points += int(finite.sum())
            dropped_points += int((~finite).sum())
            if finite.any():
                errors.append(float(np.nanmedian(result.reprojection_error[finite])))
            for camera in poses:
                frames_per_camera[camera] += 1
            frame_instances.append({
                "keypoints_3d": np.nan_to_num(result.points_3d, nan=0.0).round(4).tolist(),
                "valid": finite.tolist(),
                "confidence": np.nan_to_num(result.confidence, nan=0.0).round(4).tolist(),
                "reprojection_error": np.nan_to_num(result.reprojection_error, nan=0.0).round(3).tolist(),
                "views": sorted(poses),
            })
        frames.append({"frame": frame_index, "instances": frame_instances})

    target = Path(output_dir or project / "results" / "pose3d")
    target.mkdir(parents=True, exist_ok=True)
    _atomic_write(target / "poses3d.json", {
        "schema": SCHEMA,
        "world_unit": rig.world_unit,
        "frames": frames,
    })

    filtered_path = None
    if pose_filter:
        from animalposetracker.postprocessing.filters import filter_pose_3d

        method = str(pose_filter.get("method") or "none")
        if method != "none":
            filtered_path = target / "poses3d-filter.json"
            _atomic_write(filtered_path, _filtered_payload(frames, rig.world_unit, pose_filter))

    report = {
        "schema": SCHEMA,
        "created_by": "animalposetracker",
        "association": association,
        "world_unit": rig.world_unit,
        "calibration": {
            "path": str(rig_path),
            "cameras": {camera: {"image_size": list(c.image_size or []),
                                 "distortion": rig.distortion[camera][0]}
                        for camera, c in rig.cameras.items()},
            "snapshot": _read_yaml(rig_path),
        },
        "inputs": {camera: {"path": str(path), "frames": frame_counts[camera]}
                   for camera, path in view_paths.items()},
        "thresholds": {
            "confidence_threshold": float(confidence_threshold),
            "reprojection_threshold": float(reprojection_threshold),
            "min_views": int(min_views),
        },
        "filter": dict(pose_filter) if filtered_path else None,
        "summary": {
            "frames": len(frames),
            "instances": int(sum(len(frame["instances"]) for frame in frames)),
            "points": total_points,
            "dropped_points": dropped_points,
            "dropped_ratio": round(dropped_points / max(total_points + dropped_points, 1), 4),
            "reprojection_median_px": round(float(np.median(errors)), 3) if errors else None,
            "reprojection_p90_px": round(float(np.percentile(errors, 90)), 3) if errors else None,
            "instances_per_camera": frames_per_camera,
        },
        "outputs": {
            "poses3d": "poses3d.json",
            "poses3d_filtered": "poses3d-filter.json" if filtered_path else None,
        },
    }
    _atomic_write(target / "report.json", report)
    return report


def _filtered_payload(frames: Iterable[Mapping[str, Any]], world_unit: str,
                      pose_filter: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the shared 3D filter over the triangulated sequence, keeping the layout."""
    from animalposetracker.postprocessing.filters import filter_pose_3d

    frames = list(frames)
    if not frames:
        return {"schema": SCHEMA, "world_unit": world_unit, "frames": []}
    instance_count = max(len(frame["instances"]) for frame in frames)
    result_frames = [{"frame": frame["frame"], "instances": []} for frame in frames]
    for instance_index in range(instance_count):
        sequence = []
        keep = []
        for position, frame in enumerate(frames):
            if instance_index < len(frame["instances"]):
                instance = frame["instances"][instance_index]
                points = np.asarray(instance["keypoints_3d"], dtype=np.float64)
                confidence = np.asarray(instance["confidence"], dtype=np.float64).reshape(-1, 1)
                sequence.append(np.concatenate([points, confidence], axis=1))
                keep.append(position)
        if len(sequence) < 3:
            for position in keep:
                result_frames[position]["instances"].append(frames[position]["instances"][instance_index])
            continue
        smoothed = filter_pose_3d(
            np.stack(sequence),
            method=str(pose_filter.get("method") or "median"),
            window_length=int(pose_filter.get("window_length") or 5),
            polyorder=int(pose_filter.get("polyorder") or 2),
            confidence_threshold=pose_filter.get("confidence_threshold"),
            max_gap=int(pose_filter.get("max_gap") or 0),
        )
        for offset, position in enumerate(keep):
            source = frames[position]["instances"][instance_index]
            result_frames[position]["instances"].append({
                "keypoints_3d": smoothed[offset, :, :3].round(4).tolist(),
                "valid": source["valid"],
                "confidence": source["confidence"],
                "reprojection_error": source["reprojection_error"],
                "views": source["views"],
            })
    return {"schema": SCHEMA, "world_unit": world_unit, "frames": result_frames}


__all__ = ["SCHEMA", "ASSOCIATIONS", "reconstruct_project"]
