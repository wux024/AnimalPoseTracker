"""Render a triangulated sequence: back onto the cameras, or as a 3D turntable.

The 3D stage writes numbers; these two renderers turn them into something you can watch.
``render_overlay`` projects every 3D keypoint through the calibration onto the camera that
recorded it, so a correct rig makes the skeleton sit exactly on the animal -- a wrong rig
shows it sliding off, which no report column makes as obvious. ``render_space`` draws the
skeletons in world space and revolves around them, for figures and talks.

Both take the ``poses3d.json`` payload and write one video per call; nothing here reads or
writes the project's configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration import CameraRig

#: Colours handed out per instance, in BGR (OpenCV writes them straight into the frame).
INSTANCE_COLORS = (
    (80, 220, 80), (80, 140, 255), (255, 160, 60), (200, 80, 220),
    (60, 220, 220), (220, 220, 80), (160, 120, 255), (255, 80, 120),
)


def load_skeleton(project_dir: Path | str) -> List[Tuple[int, int]]:
    """Keypoint connections declared in ``configs/dataset.yaml`` (empty when absent)."""
    path = Path(project_dir).expanduser() / "configs" / "dataset.yaml"
    if not path.is_file():
        return []
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    skeleton = payload.get("skeleton") if isinstance(payload, Mapping) else None
    if not isinstance(skeleton, (list, tuple)):
        return []
    edges: List[Tuple[int, int]] = []
    for edge in skeleton:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            edges.append((int(edge[0]), int(edge[1])))
        except (TypeError, ValueError):
            continue
    return edges


def _instance_points(instance: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(instance.get("keypoints_3d") or [], dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(instance.get("valid") or [], dtype=bool).reshape(-1)
    if valid.size != len(points):
        valid = np.isfinite(points).all(axis=1)
    return points, valid


def _draw_skeleton(
    image: np.ndarray,
    pixels: np.ndarray,
    visible: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    """Draw the edges first, then the joints, so the joints stay readable."""
    import cv2

    height, width = image.shape[:2]

    def inside(point) -> bool:
        return 0 <= point[0] < width and 0 <= point[1] < height

    if edges:
        for first, second in edges:
            if not (visible[first] and visible[second]):
                continue
            a = tuple(np.round(pixels[first]).astype(int))
            b = tuple(np.round(pixels[second]).astype(int))
            if inside(a) and inside(b):
                cv2.line(image, a, b, color, 2, cv2.LINE_AA)
    for index in range(len(pixels)):
        if not visible[index]:
            continue
        centre = tuple(np.round(pixels[index]).astype(int))
        if inside(centre):
            cv2.circle(image, centre, 3, color, -1, cv2.LINE_AA)


def _tile(frames: Sequence[np.ndarray], max_width: int, layout: str) -> np.ndarray:
    """Scale every view to the same width and stack them side by side (or one above another)."""
    import cv2

    scaled = []
    for frame in frames:
        height, width = frame.shape[:2]
        if width > max_width:
            ratio = max_width / float(width)
            frame = cv2.resize(frame, (max_width, int(round(height * ratio))),
                               interpolation=cv2.INTER_AREA)
        scaled.append(frame)
    if not scaled:
        raise ValueError("No frames to tile")
    if layout == "vertical":
        width = max(frame.shape[1] for frame in scaled)
        padded = []
        for frame in scaled:
            if frame.shape[1] != width:
                canvas = np.zeros((frame.shape[0], width, 3), dtype=np.uint8)
                canvas[:, :frame.shape[1]] = frame
                frame = canvas
            padded.append(frame)
        return np.vstack(padded)
    height = max(frame.shape[0] for frame in scaled)
    padded = []
    for frame in scaled:
        if frame.shape[0] != height:
            canvas = np.zeros((height, frame.shape[1], 3), dtype=np.uint8)
            canvas[:frame.shape[0], :] = frame
            frame = canvas
        padded.append(frame)
    return np.hstack(padded)


class _SequentialReader:
    """Read one video forward, grabbing past frames until the requested index is reached."""

    def __init__(self, path: Path):
        import cv2

        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")
        self.position = 0    # index of the next frame `read` will return

    def frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0:
            return None
        while self.position < index:
            if not self.capture.grab():
                return None
            self.position += 1
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.position += 1
        return frame

    def release(self) -> None:
        self.capture.release()


def _open_writer(path: Path, fps: float, size: Tuple[int, int]):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise ValueError(f"Cannot write video: {path}")
    return writer


def _write_still(directory: Optional[Path], index: int, frame: np.ndarray) -> None:
    """Keep a JPEG next to the video when asked, numbered like the frame it came from."""
    import cv2

    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"frame_{index:05d}.jpg"), frame)


def render_overlay(
    payload: Mapping[str, Any],
    rig: CameraRig,
    sources: Mapping[str, Path],
    output: Path | str,
    *,
    skeleton: Sequence[Tuple[int, int]] = (),
    fps: float = 30.0,
    stride: int = 1,
    layout: str = "horizontal",
    max_width: int = 640,
    images_dir: Optional[Path] = None,
) -> int:
    """Project the 3D keypoints back onto each camera's video and tile them into one file.

    ``sources`` maps a camera to the video the 2D stage read. The camera's declared frame
    offset is applied the same way the triangulation applied it, so frame *i* of the output
    is frame *i + offset* of that camera. ``images_dir`` additionally keeps one JPEG per
    written frame. Returns the number of frames written.
    """
    if int(stride) < 1:
        raise ValueError("stride must be at least 1")
    cameras = [camera for camera in rig.cameras if camera in sources]
    if not cameras:
        raise ValueError("render_overlay needs at least one camera with a source video")

    frames = list(payload.get("frames") or [])
    if not frames:
        raise ValueError("poses3d payload holds no frames")

    readers = {camera: _SequentialReader(Path(sources[camera])) for camera in cameras}
    writer = None
    written = 0
    try:
        for step, frame_record in enumerate(frames):
            if step % int(stride):
                continue
            index = int(frame_record.get("frame", step))
            tiles = []
            for camera in cameras:
                offset = int(rig.frame_offsets.get(camera, 0))
                image = readers[camera].frame(index + offset)
                if image is None:
                    break
                image = image.copy()
                for position, instance in enumerate(frame_record.get("instances") or []):
                    if camera not in (instance.get("views") or []):
                        continue
                    points, valid = _instance_points(instance)
                    if not len(points):
                        continue
                    pixels = rig.cameras[camera].project_points(points)
                    visible = valid & np.isfinite(pixels).all(axis=1)
                    _draw_skeleton(image, pixels, visible, skeleton,
                                   INSTANCE_COLORS[position % len(INSTANCE_COLORS)])
                tiles.append(image)
            if len(tiles) != len(cameras):
                break
            canvas = _tile(tiles, int(max_width), layout)
            if writer is None:
                writer = _open_writer(Path(output), fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            _write_still(images_dir, written, canvas)
            written += 1
    finally:
        for reader in readers.values():
            reader.release()
        if writer is not None:
            writer.release()
    if writer is None:
        raise ValueError("No frames were written; check the source videos and frame offsets")
    return written


def render_space(
    payload: Mapping[str, Any],
    output: Path | str,
    *,
    skeleton: Sequence[Tuple[int, int]] = (),
    fps: float = 30.0,
    stride: int = 1,
    width: int = 960,
    height: int = 720,
    revolve: bool = True,
    degrees_per_frame: float = 1.5,
    images_dir: Optional[Path] = None,
) -> int:
    """Draw the skeletons in world space and revolve the camera around them.

    The axis limits come from the whole sequence, so the animal does not jitter while the
    plot rescales. ``images_dir`` additionally keeps one JPEG per written frame. Returns the
    number of frames written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if int(stride) < 1:
        raise ValueError("stride must be at least 1")
    frames = list(payload.get("frames") or [])
    if not frames:
        raise ValueError("poses3d payload holds no frames")

    collected = []
    for frame_record in frames:
        instances = []
        for instance in frame_record.get("instances") or []:
            points, valid = _instance_points(instance)
            if len(points) and valid.any():
                instances.append((points, valid))
        collected.append(instances)

    stacked = [points[valid] for frame in collected for points, valid in frame]
    if not stacked:
        raise ValueError("poses3d payload holds no valid 3D points")
    everything = np.vstack(stacked)
    limits = []
    for axis in range(3):
        low, high = float(everything[:, axis].min()), float(everything[:, axis].max())
        span = max(high - low, 1e-6)
        pad = span * 0.1
        limits.append((low - pad, high + pad))

    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axes = figure.add_subplot(111, projection="3d")
    # Equal box aspect: a stick figure must not look stretched because x spans more than z.
    spans = tuple(limits[axis][1] - limits[axis][0] for axis in range(3))
    writer = None
    written = 0
    try:
        for step, instances in enumerate(collected):
            if step % int(stride):
                continue
            axes.cla()
            axes.set_box_aspect(spans)
            axes.set_xlim(*limits[0])
            axes.set_ylim(*limits[1])
            axes.set_zlim(*limits[2])
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            axes.set_zlabel("z")
            if revolve:
                axes.view_init(elev=20.0, azim=(step * float(degrees_per_frame)) % 360.0)
            for position, (points, valid) in enumerate(instances):
                color = np.asarray(INSTANCE_COLORS[position % len(INSTANCE_COLORS)][::-1]) / 255.0
                for first, second in skeleton:
                    if valid[first] and valid[second]:
                        axes.plot(
                            [points[first, 0], points[second, 0]],
                            [points[first, 1], points[second, 1]],
                            [points[first, 2], points[second, 2]],
                            color=color, linewidth=2.0,
                        )
                axes.scatter(points[valid, 0], points[valid, 1], points[valid, 2],
                             color=color, s=12, depthshade=False)
            figure.canvas.draw()
            buffer = np.asarray(figure.canvas.buffer_rgba())
            frame = buffer[:, :, :3][:, :, ::-1].copy()   # RGBA -> BGR
            if writer is None:
                writer = _open_writer(Path(output), fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            _write_still(images_dir, written, frame)
            written += 1
    finally:
        plt.close(figure)
        if writer is not None:
            writer.release()
    if writer is None:
        raise ValueError("No frames were written; the sequence is empty after striding")
    return written


def source_videos(
    predictions_by_camera: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Path]:
    """The video each camera's prediction records came from.

    ``predictions.json`` stores one summary record per video (``source`` plus ``output``);
    per-frame records carry ``source`` and ``frame``. Either shape names the video.
    """
    sources: Dict[str, Path] = {}
    for camera, records in predictions_by_camera.items():
        for record in records:
            source = record.get("source")
            if source and str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".m4v")):
                sources[camera] = Path(source)
                break
    return sources


__all__ = [
    "INSTANCE_COLORS",
    "load_skeleton",
    "render_overlay",
    "render_space",
    "source_videos",
]
