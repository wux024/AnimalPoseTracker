"""Multi-view reconstruction for a project: 2D predictions in, ``results/pose3d`` out.

The runner owns the whole 3D stage: it reads one ``predictions.json`` per
camera, associates instances across views (``index`` or ``epipolar``), undistorts
the pixels with the rig's distortion model, triangulates, and writes the files
agreed for the 3D protocol: ``poses3d.json``, ``poses3d-filter.json`` (only when
filtering is enabled) and ``report.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .associate import associate_instances
from .calibration import CameraRig, load_camera_rig
from .core import triangulate_groups

SCHEMA = "pose3d/1"
ASSOCIATIONS = {"index", "epipolar"}


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
    # A predictions file holds either one record per image or one record per video frame
    # (``source`` + ``frame`` + ``predictions``); summary records carry no predictions and
    # are dropped. The source may therefore be a folder of images or a video.
    records = [record for record in payload
               if isinstance(record, Mapping) and record.get("predictions") is not None]
    if not records:
        raise ValueError(f"{path} holds no prediction records")
    if all("frame" in record for record in records):
        records.sort(key=lambda record: int(record["frame"]))
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


def _index_groups(
    per_view: Sequence[tuple], min_views: int
) -> List[Dict[str, int]]:
    """Pair the n-th instance of every camera with the n-th of the others.

    Detection order is only an identity when a frame holds a single animal, or
    when every camera happens to order them the same way; that is why the
    ``epipolar`` association exists.
    """
    instance_count = max((len(instances) for _, instances in per_view), default=0)
    groups: List[Dict[str, int]] = []
    for instance_index in range(instance_count):
        group = {camera: instance_index for camera, instances in per_view
                 if instance_index < len(instances)}
        if len(group) >= int(min_views):
            groups.append(group)
    return groups


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
    max_sampson_px: float = 25.0,
    pose_filter: Optional[Mapping[str, Any]] = None,
    check_sync: bool = False,
    backend: str = "numpy",
    refine: bool = True,
) -> Dict[str, Any]:
    """Triangulate every frame of a 3D-enabled project and write the protocol files.

    ``backend`` picks the batched solver from
    :mod:`animalposetracker.pose3d.accelerate` and is left to the caller: ``numpy``
    is the default and needs no extra install; ``numba`` and ``jax-gpu`` are there
    when a block is large enough to pay for them. ``refine`` keeps the per-joint
    least-squares polish; turning it off is what makes the batched path several
    times faster, at the cost of the last fraction of a pixel.
    """
    project = Path(project_dir).expanduser().resolve()
    identity = _read_yaml(project / "project.yaml") if (project / "project.yaml").is_file() else {}
    if not identity:
        raise FileNotFoundError(f"Not a project directory (no project.yaml): {project}")
    if identity.get("public_dataset"):
        raise ValueError(
            f"Project '{project.name}' was created from the public benchmark "
            f"'{identity['public_dataset']}'; it has no multi-camera capture, so 3D is not available"
        )
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
    if max(frame_counts.values()) - min(frame_counts.values()) > 60:
        raise ValueError(f"Views disagree on frame count: {frame_counts}")
    anchor = next(iter(rig.cameras))
    offsets = {camera: int(rig.frame_offsets.get(camera, 0)) for camera in rig.cameras}

    sync_check = None
    if check_sync:
        from .sync import estimate_frame_offsets

        series = {}
        for view in loaded:
            frames_series = []
            for record in view.records:
                instances = _instances(record)
                if not instances:
                    break
                frames_series.append(rig.undistort(view.camera, instances[0]))
            series[view.camera] = frames_series
        sync_check = estimate_frame_offsets(rig, series)
        for camera, curve in sync_check["curves"].items():
            best = curve["best_offset"]
            if not curve["confident"] or best == offsets.get(camera, 0):
                continue
            raise ValueError(
                f"Camera '{camera}' needs a frame offset of {best} (epipolar median "
                f"{curve['best_median_px']} px there vs {curve['zero_offset_median_px']} px at "
                f"offset 0). Set sync.offsets.{camera}: {best} in configs/cameras.yaml"
            )

    frames: List[Dict[str, Any]] = []
    frames_per_camera = {camera: 0 for camera in rig.cameras}
    total_points = 0
    dropped_points = 0
    errors: List[float] = []
    unmatched_instances = {camera: 0 for camera in rig.cameras}
    matched_instances = 0

    total_frames = frame_counts[anchor]
    # Frames are handled in blocks: association fills a block, one triangulate_groups
    # call solves the whole block, then the block becomes output. A backend sees every
    # frame in the block at once, which is what makes numba and jax worth naming, and
    # the block keeps memory flat on recordings that run for tens of thousands of frames.
    block_size = 2048
    for block_start in tqdm(range(0, total_frames, block_size), desc="triangulate",
                            unit="block", disable=None, dynamic_ncols=True, leave=True):
        block_end = min(block_start + block_size, total_frames)
        pending: List[Dict[str, np.ndarray]] = []
        layout: List[Tuple[int, List[Tuple[Dict[str, int], int]]]] = []
        for frame_index in range(block_start, block_end):
            per_view = [(view.camera,
                         [rig.undistort(view.camera, instance)
                          for instance in (_instances(view.records[frame_index + offsets[view.camera]])
                                           if 0 <= frame_index + offsets[view.camera] < len(view.records) else [])])
                        for view in loaded]
            view_map = dict(per_view)
            if association == "index":
                groups = _index_groups(per_view, min_views)
            else:
                # Matching is more sensitive to noise than triangulation: a wrong
                # identity cannot be repaired later, so keep the joint keypoints at
                # least reasonably confident even when the caller triangulates loose ones.
                groups = associate_instances(
                    view_map, rig,
                    max_sampson_px=max_sampson_px,
                    confidence_threshold=max(float(confidence_threshold), 0.1),
                    min_views=min_views,
                )
                matched_instances += sum(len(group) for group in groups)
                for camera, instances in per_view:
                    used = sum(1 for group in groups if camera in group)
                    unmatched_instances[camera] += len(instances) - used

            slots: List[Tuple[Dict[str, int], int]] = []
            for group in groups:
                slots.append((group, len(pending)))
                pending.append({camera: view_map[camera][index]
                                for camera, index in group.items()})
            layout.append((frame_index, slots))

        solved = triangulate_groups(
            pending, rig.cameras,
            confidence_threshold=confidence_threshold,
            reprojection_threshold=reprojection_threshold,
            min_views=min_views,
            refine=refine,
            backend=backend,
        )

        for frame_index, slots in layout:
            frame_instances: List[Dict[str, Any]] = []
            for group, slot in slots:
                result = solved[slot]
                poses = pending[slot]
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
                    "view_indices": {camera: int(index) for camera, index in sorted(group.items())},
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
        "association_stats": {
            "max_sampson_px": float(max_sampson_px),
            "matched_instances": int(matched_instances),
            "unmatched_instances": dict(unmatched_instances),
        } if association == "epipolar" else None,
        "triangulation": {
            "backend": backend,
            "refine": bool(refine),
            "reject_outliers": True,
        },
        "world_unit": rig.world_unit,
        "sync": {"mode": rig.sync_mode, "declared_offsets": offsets, "check": sync_check},
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
