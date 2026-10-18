"""Render the demo clips for the landing page from the DataVisualizationGUI experiments.

Each clip is a three-panel animation: the source video on the left, the reconstructed
3D skeleton on the right, and the XY track underneath. The data layout is the one the
old analysis GUI used — one row per frame, six columns per keypoint
(``<name>_x``, ``_y``, ``_z``, ``_error``, ``_ncams``, ``_score``).

    python tools/demo_clips.py hand koi pendulum
    python tools/demo_clips.py hand --frames 120     # quick preview

Output goes to ``AnimalPoseTrackerProject/demos`` as GIF (and a matching MP4).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SOURCE = Path("C:/Users/wux024/project/AdamPosePlug")
OUTPUT = Path("C:/Users/wux024/project/AnimalPoseTrackerProject/demos")

# name -> csv, video, ordered keypoints, skeleton edges
# The CSV and the video must hold the same number of frames; the renderer refuses to
# guess an offset, because a silent trim is exactly how the clips went out of sync.
SPECS: dict[str, dict] = {
    "hand": {
        "csv": "pose3d-hand/hand_0_cam.csv",
        "video": "video3d-hand/hand_0_cam.mp4",
        "points": ["base"]
        + [f"{part}{finger}" for finger in range(1, 6)
           for part in ("MCP", "PIP", "DIP", "tip")],
        "chains": [["base"] + [f"{part}{f}" for part in ("MCP", "PIP", "DIP", "tip")]
                   for f in range(1, 6)],
    },
    "koi": {
        "csv": "pose-3d-0512/koipose3.csv",
        "video": "video3d-0512/koipose3.mp4",
        "points": ["fish_head", "dorsal_fin_0", "dorsal_fin_1", "dorsal_fin_2",
                   "dorsal_fin_3", "fish_body_r", "fish_body_l", "fish_eye_r",
                   "pectoral_tail_root_r", "pectoral_tail_up_r", "pectoral_tail_middle_r",
                   "pectoral_tail_down_r", "fish_eye_l", "pectoral_tail_root_l",
                   "pectoral_tail_up_l", "pectoral_tail_middle_l", "pectoral_tail_down_l",
                   "fish_tail", "tail_fin_up", "tail_fin_middle", "tail_fin_down"],
        "chains": [
            ["fish_head", "dorsal_fin_0", "dorsal_fin_1", "dorsal_fin_2", "dorsal_fin_3"],
            ["fish_head", "fish_body_r", "pectoral_tail_root_r", "fish_tail"],
            ["fish_head", "fish_body_l", "pectoral_tail_root_l", "fish_tail"],
            ["pectoral_tail_root_r", "pectoral_tail_up_r", "pectoral_tail_middle_r",
             "pectoral_tail_down_r"],
            ["pectoral_tail_root_l", "pectoral_tail_up_l", "pectoral_tail_middle_l",
             "pectoral_tail_down_l"],
            ["fish_tail", "tail_fin_up", "tail_fin_middle", "tail_fin_down"],
        ],
    },
    "pendulum": {
        "csv": "pose-3d-ChaoticPendulum/adam0_30fps_cam.csv",
        "video": "videos-3d-ChaoticPendulum/adam0_30fps_cam.mp4",
        "points": ["big_up", "big_down", "small_up", "small_down"],
        "chains": [["big_up", "big_down", "small_up", "small_down"]],
    },
}


def load_frames(path: Path, names: list[str]) -> np.ndarray:
    """(frames, keypoints, 3) array of xyz."""
    series = {name: [] for name in names}
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            for name in names:
                try:
                    series[name].append([float(row[f"{name}_x"]), float(row[f"{name}_y"]),
                                         float(row[f"{name}_z"])])
                except (KeyError, TypeError, ValueError):
                    series[name].append([np.nan, np.nan, np.nan])
    return np.stack([np.array(series[name], dtype=float) for name in names], axis=1)


def render(name: str, spec: dict, limit: int | None, fps: int, stride: int) -> None:
    points = spec["points"]
    data = load_frames(SOURCE / spec["csv"], points)
    cap = cv2.VideoCapture(str(SOURCE / spec["video"]))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    if video_frames and len(data) != video_frames:
        raise ValueError(
            f"{name}: {len(data)} pose rows against {video_frames} video frames — "
            "the two must match frame for frame, otherwise the skeleton drifts"
        )
    total = len(data)
    if limit:
        total = min(total, limit)

    index = {name: i for i, name in enumerate(points)}
    finite = data[np.isfinite(data).all(axis=2)]
    lo = finite.min(axis=0) if len(finite) else np.zeros(3)
    hi = finite.max(axis=0) if len(finite) else np.ones(3)
    centre = (lo + hi) / 2

    # AdamPosePlug stacks the two panels: 2D on top, 3D underneath, both 1640x451
    # (PoseEstimation2D at y=150 and PoseEstimation3D at y=609 in a 1920x1080 window).
    figure = plt.figure(figsize=(16.4, 9.0), dpi=100)
    axis_video = figure.add_axes([0.0, 0.5, 1.0, 0.5])
    axis_3d = figure.add_axes([0.0, 0.0, 1.0, 0.5], projection="3d")
    axis_video.set_axis_off()

    staging = Path(tempfile.mkdtemp(prefix=f"apt-{name}-"))
    palette = [tuple(c) for c in plt.cm.tab20(np.linspace(0, 1, len(points)))]
    try:
        for frame in range(total):
            ok, image = cap.read()
            if frame % stride:
                continue

            # --- 2DPose: crop 580 px off the bottom, then scale to 450 tall,
            # exactly as AdamPosePlugMain.videoframe does ---
            axis_video.clear()
            axis_video.set_axis_off()
            if ok:
                cut = image.shape[0] - 580
                image = image[:cut, :, :]
                width = int(image.shape[1] / image.shape[0] * 450)
                image = cv2.resize(image, (width, 450))
                axis_video.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # --- 3DPose: no axis, no limits, arrows anchored at (-150, -80, -200),
            # z negated; matplotlib autoscales just like the original ---
            axis_3d.clear()
            axis_3d.set_axis_off()
            xyz = data[frame].copy()
            xyz[:, 2] = -xyz[:, 2]
            for chain in spec["chains"]:
                idx = [index[p] for p in chain]
                axis_3d.plot(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], "b-")
            axis_3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=100,
                            c=[palette[i] for i in range(len(points))], depthshade=False)
            dx, dy, dz = -150.0, -80.0, -200.0
            for direction, colour in (((1, 0, 0), "r"), ((0, 1, 0), "g"), ((0, 0, 1), "y")):
                axis_3d.quiver(dx, dy, dz, *direction, length=60, normalize=True,
                               color=colour, arrow_length_ratio=0.12)
            labels = {"family": "Times New Roman", "size": 14, "style": "italic"}
            axis_3d.text(dx + 65, dy, dz, "x", fontdict=labels)
            axis_3d.text(dx, dy + 65, dz, "y", fontdict=labels)
            axis_3d.text(dx, dy, dz + 65, "z", fontdict=labels)
            figure.savefig(staging / f"{frame // stride:04d}.png", facecolor="white")
        cap.release()

        OUTPUT.mkdir(parents=True, exist_ok=True)
        gif = OUTPUT / f"{name}.gif"
        mp4 = OUTPUT / f"{name}.mp4"
        palette = "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(staging / "%04d.png"), "-vf",
             f"fps={fps},scale=760:-1:flags=lanczos,{palette}", str(gif)],
            check=True,
        )
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", str(staging / "%04d.png"), "-vf", "scale=760:-1:flags=lanczos",
                 "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
                check=True,
            )
        except subprocess.CalledProcessError:
            # the GIF is the deliverable; an unavailable x264 build is not fatal
            mp4.unlink(missing_ok=True)
            print(f"{name}: mp4 skipped (no usable encoder)")
        detail = f"{gif.stat().st_size // 1024} KB"
        if mp4.exists():
            detail += f", {mp4.name} {mp4.stat().st_size // 1024} KB"
        print(f"{name}: {total} frames -> {gif.name} ({detail})")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(SPECS), choices=list(SPECS))
    parser.add_argument("--frames", type=int, default=None, help="render only the first N")
    parser.add_argument("--fps", type=int, default=15, help="output frame rate")
    parser.add_argument("--stride", type=int, default=2, help="keep one frame in N")
    args = parser.parse_args()
    for name in args.names or list(SPECS):
        render(name, SPECS[name], args.frames, args.fps, max(1, args.stride))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
