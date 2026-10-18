"""Turn a ``poses3d.json`` payload into analysis-ready files.

The 3D stage writes one JSON document with nested frames and instances.  Most
downstream work (statistics, plotting, spreadsheets) wants a flat table, so this
module provides three views of the same data:

* a CSV with one row per (frame, instance, keypoint) and the confidence, validity,
  reprojection error, and view count;
* a JSON summary with frame/instance/point counts, dropped ratio, reprojection
  percentiles, per-keypoint coverage, and the world-space bounds;
* an optional PNG with XY, XZ, and YZ trajectories for one instance;
* optional videos: the sequence projected back onto each camera, and a revolving 3D view.

The CLI command is ``pose3d-render`` (``export`` alone is the model exporter, and two
commands called export invited exactly the confusion this rename removed).

Nothing here imports the inference or training stacks: the command only reads
the JSON protocol and writes plain files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["load_poses", "iter_rows", "summarize", "write_csv", "write_plot", "run"]

COLUMNS = [
    "frame", "instance", "keypoint", "x", "y", "z",
    "confidence", "valid", "reprojection_error", "num_views",
]


def load_poses(path: Path) -> Dict[str, object]:
    """Read and minimally validate a ``poses3d.json`` document."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("frames"), list):
        raise ValueError(f"{path} is not a pose3d payload (missing a 'frames' list)")
    for frame in payload["frames"]:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("instances", []), list):
            raise ValueError(f"{path} has a malformed frame entry")
    return dict(payload)


def _point_rows(payload: Mapping[str, object], include_invalid: bool) -> Iterable[Dict[str, object]]:
    for frame in payload["frames"]:
        frame_index = int(frame.get("frame", 0))
        for instance_index, instance in enumerate(frame.get("instances") or []):
            points = np.asarray(instance.get("keypoints_3d", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                continue
            count = len(points)
            valid = np.asarray(instance.get("valid", [True] * count), dtype=bool)
            confidence = np.asarray(instance.get("confidence", [0.0] * count), dtype=np.float64)
            reprojection = np.asarray(instance.get("reprojection_error", [0.0] * count), dtype=np.float64)
            views = instance.get("views") or []
            for keypoint in range(count):
                if not include_invalid and not bool(valid[keypoint]):
                    continue
                x, y, z = (float(value) for value in points[keypoint])
                yield {
                    "frame": frame_index,
                    "instance": instance_index,
                    "keypoint": keypoint,
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": float(confidence[keypoint]) if keypoint < len(confidence) else 0.0,
                    "valid": bool(valid[keypoint]),
                    "reprojection_error": (
                        float(reprojection[keypoint]) if keypoint < len(reprojection) else 0.0
                    ),
                    "num_views": len(views),
                }


def iter_rows(payload: Mapping[str, object], include_invalid: bool = False) -> Iterable[Dict[str, object]]:
    """Yield the flat rows, skipping invalid points unless asked otherwise."""
    yield from _point_rows(payload, include_invalid)


def summarize(payload: Mapping[str, object]) -> Dict[str, object]:
    """Counts, reprojection percentiles, keypoint coverage, and world bounds."""
    frames = list(payload.get("frames") or [])
    instances = 0
    valid_points = 0
    invalid_points = 0
    errors: List[float] = []
    coverage: Dict[int, List[int]] = {}
    bounds_min = np.full(3, np.inf)
    bounds_max = np.full(3, -np.inf)
    per_frame_instances: List[int] = []
    for frame in frames:
        frame_instances = list(frame.get("instances") or [])
        per_frame_instances.append(len(frame_instances))
        instances += len(frame_instances)
        for instance in frame_instances:
            points = np.asarray(instance.get("keypoints_3d", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                continue
            count = len(points)
            valid = np.asarray(instance.get("valid", [True] * count), dtype=bool)
            reprojection = np.asarray(instance.get("reprojection_error", [0.0] * count), dtype=np.float64)
            for keypoint in range(count):
                coverage.setdefault(keypoint, [0, 0])[1] += 1
                if valid[keypoint]:
                    valid_points += 1
                    coverage[keypoint][0] += 1
                    point = points[keypoint]
                    if np.isfinite(point).all():
                        bounds_min = np.minimum(bounds_min, point)
                        bounds_max = np.maximum(bounds_max, point)
                    if keypoint < len(reprojection) and np.isfinite(reprojection[keypoint]):
                        errors.append(float(reprojection[keypoint]))
                else:
                    invalid_points += 1
    total = valid_points + invalid_points
    return {
        "schema": payload.get("schema"),
        "world_unit": payload.get("world_unit", "mm"),
        "frames": len(frames),
        "instances": instances,
        "instances_per_frame": {
            "min": min(per_frame_instances) if per_frame_instances else 0,
            "max": max(per_frame_instances) if per_frame_instances else 0,
        },
        "points": valid_points,
        "dropped_points": invalid_points,
        "dropped_ratio": round(invalid_points / total, 6) if total else 0.0,
        "reprojection_median_px": round(float(np.median(errors)), 6) if errors else None,
        "reprojection_p90_px": round(float(np.percentile(errors, 90)), 6) if errors else None,
        "keypoints": {
            str(keypoint): {"valid": values[0], "total": values[1]}
            for keypoint, values in sorted(coverage.items())
        },
        "bounds": {
            "min": [round(float(value), 4) for value in bounds_min] if np.isfinite(bounds_min).all() else None,
            "max": [round(float(value), 4) for value in bounds_max] if np.isfinite(bounds_max).all() else None,
        },
    }


def write_csv(payload: Mapping[str, object], path: Path, include_invalid: bool = False) -> int:
    """Write the flat table and return the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in iter_rows(payload, include_invalid):
            writer.writerow(row)
            count += 1
    return count


def write_plot(payload: Mapping[str, object], path: Path, instance: int = 0) -> None:
    """Write XY, XZ, and YZ trajectories for one instance to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectories: Dict[int, List[np.ndarray]] = {}
    for frame in payload.get("frames") or []:
        instances = frame.get("instances") or []
        if instance >= len(instances):
            continue
        entry = instances[instance]
        points = np.asarray(entry.get("keypoints_3d", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        valid = np.asarray(entry.get("valid", [True] * len(points)), dtype=bool)
        for keypoint in range(len(points)):
            if valid[keypoint] and np.isfinite(points[keypoint]).all():
                trajectories.setdefault(keypoint, []).append(points[keypoint])
    if not trajectories:
        raise ValueError(f"instance {instance} has no valid 3D points to plot")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(trajectories), 1)))
    for color, (keypoint, points) in zip(colors, sorted(trajectories.items())):
        values = np.asarray(points)
        axes[0].plot(values[:, 0], values[:, 1], color=color, linewidth=1.0)
        axes[1].plot(values[:, 0], values[:, 2], color=color, linewidth=1.0)
        axes[2].plot(values[:, 1], values[:, 2], color=color, linewidth=1.0)
        axes[0].scatter(values[-1:, 0], values[-1:, 1], color=color, s=8)
    unit = payload.get("world_unit", "mm")
    for axis, (horizontal, vertical) in zip(axes, (("x", "y"), ("x", "z"), ("y", "z"))):
        axis.set_xlabel(f"{horizontal} ({unit})")
        axis.set_ylabel(f"{vertical} ({unit})")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
    axes[0].set_title(f"instance {instance}: {len(trajectories)} keypoints")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli pose3d-render",
        description="Render pose3d results to CSV, a JSON summary, a trajectory plot, or a video.",
        epilog=(
            "By default the command reads <project>/results/pose3d/poses3d.json and writes "
            "poses3d.csv next to it. Use --filtered for the temporally filtered sequence."
        ),
    )
    parser.add_argument("--config", default="configs/runtime.yaml", help="project runtime config")
    parser.add_argument("--poses", help="explicit path to poses3d.json or poses3d-filter.json")
    parser.add_argument("--filtered", action="store_true",
                        help="read results/pose3d/poses3d-filter.json instead of poses3d.json")
    parser.add_argument("--output", help="CSV path (default: next to the input, .csv)")
    parser.add_argument("--include-invalid", action="store_true", help="keep invalid keypoints in the CSV")
    parser.add_argument("--summary", action="store_true", help="print the JSON summary to stdout")
    parser.add_argument("--summary-output", help="write the JSON summary to this path")
    parser.add_argument("--plot", help="write a trajectory PNG for one instance")
    parser.add_argument("--instance", type=int, default=0, help="instance index used by --plot")
    parser.add_argument("--video", default="none", choices=["none", "overlay", "space", "both"],
                        help="render the sequence: 'overlay' projects it back onto each "
                             "camera's video, 'space' revolves a 3D view, 'both' does both")
    parser.add_argument("--views", action="append", default=[],
                        help="camera=path to that camera's predictions.json (finds its video)")
    parser.add_argument("--calibration", help="camera calibration (default: configs/cameras.yaml)")
    parser.add_argument("--images", action="store_true",
                        help="with --video, also keep one JPEG per frame")
    parser.add_argument("--fps", type=float, default=30.0, help="video frame rate")
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    parser.add_argument("--layout", default="horizontal", choices=["horizontal", "vertical"],
                        help="how overlay tiles the cameras")
    parser.add_argument("--max-width", type=int, default=640,
                        help="width of each camera tile in overlay mode")
    parser.add_argument("--width", type=int, default=960, help="3D view width")
    parser.add_argument("--height", type=int, default=720, help="3D view height")
    parser.add_argument("--no-revolve", action="store_true",
                        help="keep the 3D view's azimuth fixed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        project = Path(args.config).expanduser().resolve().parent.parent
        if args.poses:
            poses_path = Path(args.poses).expanduser()
        else:
            name = "poses3d-filter.json" if args.filtered else "poses3d.json"
            poses_path = project / "results" / "pose3d" / name
            if args.filtered and not poses_path.is_file():
                raise FileNotFoundError(
                    f"{poses_path} does not exist; run `animalpose-cli pose3d --pose-filter ...` first"
                )
        if not poses_path.is_file():
            raise FileNotFoundError(f"pose3d payload does not exist: {poses_path}")
        payload = load_poses(poses_path)
        output = Path(args.output) if args.output else poses_path.with_suffix(".csv")
        rows = write_csv(payload, output, include_invalid=args.include_invalid)
        summary = summarize(payload)
        if args.summary:
            print(json.dumps(summary, indent=2))
        if args.summary_output:
            Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"wrote {args.summary_output}")
        if args.plot:
            write_plot(payload, Path(args.plot), instance=args.instance)
            print(f"wrote {args.plot}")
        if args.video != "none":
            for path, frames in _render_videos(payload, project, args):
                print(f"wrote {path} ({frames} frames)")
        print(
            f"pose3d-render: {rows} rows from {summary['frames']} frames, "
            f"{summary['instances']} instances ({summary['dropped_ratio'] * 100:.1f}% dropped) | "
            f"wrote {output}"
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"pose3d-render failed: {error}")
        return 1


def _view_records(project: Path, rig, views: Sequence[str]) -> dict:
    """The prediction records per camera, from --views or the project's own layout."""
    from .cli import _parse_views

    paths = _parse_views(views)
    if not paths:
        paths = {camera: project / "results" / "predict" / camera / "predictions.json"
                 for camera in rig.cameras}
    records = {}
    for camera, path in paths.items():
        path = Path(path)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("predictions") or payload.get("frames") or []
            records[camera] = [record for record in payload if isinstance(record, dict)]
    return records


def _render_videos(payload, project: Path, args) -> list:
    """Render what --video asked for; returns ``(path, frame count)`` pairs."""
    from .calibration import load_camera_rig
    from .render import load_skeleton, render_overlay, render_space, source_videos

    suffix = "-filter" if args.filtered else ""
    root = project / "results" / "pose3d"
    video_dir = root / f"videos{suffix}"
    image_dir = root / f"images{suffix}" if args.images else None
    skeleton = load_skeleton(project)
    written = []
    if args.video in ("overlay", "both"):
        rig = load_camera_rig(Path(args.calibration or project / "configs" / "cameras.yaml"))
        sources = source_videos(_view_records(project, rig, args.views))
        if not sources:
            raise ValueError(
                "No camera video found for the overlay: point --views at predictions made "
                "from videos, or render only the 3D view with --video space"
            )
        target = video_dir / "poses3d-overlay.mp4"
        written.append((target, render_overlay(
            payload, rig, sources, target, skeleton=skeleton, images_dir=image_dir,
            fps=args.fps, stride=args.stride, layout=args.layout, max_width=args.max_width,
        )))
    if args.video in ("space", "both"):
        target = video_dir / "poses3d-space.mp4"
        written.append((target, render_space(
            payload, target, skeleton=skeleton, images_dir=image_dir, fps=args.fps,
            stride=args.stride, width=args.width, height=args.height,
            revolve=not args.no_revolve,
        )))
    return written


if __name__ == "__main__":
    raise SystemExit(run())
