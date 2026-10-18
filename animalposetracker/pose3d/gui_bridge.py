"""Translate the 3D page's form values into the commands the CLI already owns.

The GUI never re-implements calibration or triangulation: it collects paths and
numbers, and this module turns them into an ``animalpose-cli`` invocation that
runs with the project directory as its working directory. Keeping the mapping
here (pure functions, no Qt) means the page's behaviour is testable without a
display, and the CLI stays the single implementation of the 3D stage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

DEFAULT_CALIBRATION = "configs/cameras.yaml"
DEFAULT_RUNTIME = "configs/runtime.yaml"


def _module_command(command: str) -> List[str]:
    """The project-relative entry point; callers run it with the project as cwd."""
    return [sys.executable, "-m", "animalposetracker.cli", command]


def camera_names(project_dir: Path | str) -> List[str]:
    """Camera ids declared in ``configs/cameras.yaml``, in file order.

    An empty template (``create --pose3d`` writes one) therefore yields an empty
    list, and the page asks the user for the rig instead of inventing names.
    """
    path = Path(project_dir).expanduser() / DEFAULT_CALIBRATION
    if not path.is_file():
        return []
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    cameras = payload.get("cameras") if isinstance(payload, Mapping) else None
    return [str(name) for name in cameras] if isinstance(cameras, Mapping) else []


def prediction_path(project_dir: Path | str, camera: str, filtered: bool = False) -> Path:
    """Where the 2D stage wrote this camera's predictions (filtered on request)."""
    directory = Path(project_dir).expanduser() / "results" / "predict" / str(camera)
    return directory / ("predictions-filter.json" if filtered else "predictions.json")


def _view_arguments(views: Mapping[str, Union[str, Path]]) -> List[str]:
    if not views:
        raise ValueError("At least one camera view is required")
    arguments: List[str] = []
    for camera, path in views.items():
        camera = str(camera).strip()
        if not camera:
            raise ValueError("Camera names must not be empty")
        if path is None or not str(path).strip():
            raise ValueError(f"Camera '{camera}' has no source path")
        arguments.extend(["--view", f"{camera}={Path(path).expanduser()}"])
    if len(views) < 2:
        raise ValueError("A multi-camera calibration needs at least two views")
    return arguments


def build_calibrate_command(
    project_dir: Path | str,
    views: Mapping[str, Union[str, Path]],
    *,
    board_squares: str = "10x7",
    square_size: float = 22.0,
    marker_size: float = 16.5,
    dictionary: str = "DICT_4X4_50",
    output: str = DEFAULT_CALIBRATION,
    force: bool = False,
    intrinsics_from: Optional[Union[str, Path]] = None,
    sync: str = "hardware",
    world_unit: str = "mm",
    frame_step: int = 0,
    max_frames: int = 300,
    intrinsic_frames: int = 60,
    report: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Command that fits one rig from per-camera ChArUco frames or videos."""
    command = _module_command("calibrate")
    command.extend(_view_arguments(views))
    command.extend(["--board-squares", str(board_squares).strip() or "10x7"])
    command.extend(["--square-size", f"{float(square_size):g}"])
    command.extend(["--marker-size", f"{float(marker_size):g}"])
    command.extend(["--dictionary", str(dictionary)])
    command.extend(["--output", str(output)])
    command.extend(["--sync", str(sync)])
    command.extend(["--world-unit", str(world_unit)])
    command.extend(["--frame-step", str(int(frame_step))])
    command.extend(["--max-frames", str(int(max_frames))])
    command.extend(["--intrinsic-frames", str(int(intrinsic_frames))])
    if int(force):
        command.append("--force")
    if intrinsics_from:
        command.extend(["--intrinsics-from", str(Path(intrinsics_from).expanduser())])
    if report:
        command.extend(["--report", str(Path(report).expanduser())])
    return command


def build_pose3d_command(
    project_dir: Path | str,
    *,
    views: Optional[Mapping[str, Union[str, Path]]] = None,
    association: str = "index",
    max_sampson: float = 25.0,
    min_views: int = 2,
    confidence_threshold: float = 0.05,
    reprojection_threshold: float = 8.0,
    check_sync: bool = False,
    pose_filter: str = "none",
    filter_window: int = 5,
    output: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Command that triangulates every camera's predictions into 3D."""
    command = _module_command("pose3d")
    command.extend(["--config", DEFAULT_RUNTIME])
    command.extend(["--association", str(association)])
    command.extend(["--max-sampson", f"{float(max_sampson):g}"])
    command.extend(["--min-views", str(int(min_views))])
    command.extend(["--confidence-threshold", f"{float(confidence_threshold):g}"])
    command.extend(["--reprojection-threshold", f"{float(reprojection_threshold):g}"])
    command.extend(["--pose-filter", str(pose_filter)])
    command.extend(["--filter-window", str(int(filter_window))])
    if check_sync:
        command.append("--check-sync")
    if output:
        command.extend(["--output", str(output)])
    for camera, path in (views or {}).items():
        command.extend(["--views", f"{camera}={Path(path).expanduser()}"])
    return command


def build_export_command(
    project_dir: Path | str,
    *,
    filtered: bool = False,
    plot: Optional[Union[str, Path]] = None,
    instance: int = 0,
    summary_output: Optional[Union[str, Path]] = None,
    include_invalid: bool = False,
    video: str = "none",
) -> List[str]:
    """Command that writes the CSV, the JSON summary, the plot and the videos."""
    command = _module_command("pose3d-render")
    command.extend(["--config", DEFAULT_RUNTIME])
    if filtered:
        command.append("--filtered")
    if plot:
        command.extend(["--plot", str(Path(plot).expanduser())])
        command.extend(["--instance", str(int(instance))])
    if summary_output:
        command.extend(["--summary-output", str(Path(summary_output).expanduser())])
    if include_invalid:
        command.append("--include-invalid")
    if video != "none":
        command.extend(["--video", str(video)])
    return command


def default_views(project_dir: Path | str, filtered: bool = False) -> Dict[str, Path]:
    """The per-camera prediction files a created project is expected to hold."""
    return {camera: prediction_path(project_dir, camera, filtered)
            for camera in camera_names(project_dir)}


__all__ = [
    "build_calibrate_command",
    "build_export_command",
    "build_pose3d_command",
    "camera_names",
    "default_views",
    "prediction_path",
]
