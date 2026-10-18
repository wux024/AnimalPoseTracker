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
from PIL import Image  # noqa: E402

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


def crop_dead_space(image: np.ndarray) -> np.ndarray:
    """Drop the flat band under the camera tiles.

    AdamPosePlug hard-codes ``frame_height - 580``; that constant is tied to the height
    of their capture, and applied to ours it cuts through the hands. Trimming to the last
    row that carries detail does the same job for any resolution.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows = np.where(gray.std(axis=1) > 1.5)[0]
    if len(rows) == 0:
        return image
    return image[: int(rows[-1]) + 1]


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

    # In AdamPosePlug these are two separate widgets (a QLabel and a FigureCanvas), so
    # they are rendered as two canvases and the frames are stacked afterwards. Sharing
    # one figure squeezed the 3D axes into a corner.
    fig_video = plt.figure(figsize=(16.4, 4.51), dpi=100)
    axis_video = fig_video.add_axes([0.0, 0.0, 1.0, 1.0])
    axis_video.set_axis_off()

    fig_3d = plt.figure(figsize=(20.5, 5.62), dpi=80)     # 1640 x 450, as in the original
    axis_3d = fig_3d.add_subplot(1, 1, 1, projection="3d")
    fig_3d.subplots_adjust(top=1, left=0, right=1, bottom=0, hspace=0, wspace=0)
    axis_3d.set_axis_off()

    staging = Path(tempfile.mkdtemp(prefix=f"apt-{name}-"))
    palette = [tuple(c) for c in plt.cm.tab20(np.linspace(0, 1, len(points)))]
    dx, dy, dz = -150.0, -80.0, -200.0

    # The reference draws the axes once in drawfigure_init and then only moves the
    # markers. Redrawing the whole axes each frame made autoscale chase the skeleton,
    # so the arrows drifted — hence one fixed frame for the whole clip.
    xyz_all = data.copy()
    xyz_all[:, :, 2] = -xyz_all[:, :, 2]

    axis_3d.set_axis_off()
    lines = []
    for chain in spec["chains"]:
        idx = [index[p] for p in chain]
        lines.append(axis_3d.plot([0], [0], [0], "b-")[0])
    markers = axis_3d.plot(xyz_all[0, :, 0], xyz_all[0, :, 1], xyz_all[0, :, 2],
                           linestyle="none", marker="o", markersize=6,
                           color="b")[0]
    for direction, colour in (((1, 0, 0), "r"), ((0, 1, 0), "g"), ((0, 0, 1), "y")):
        axis_3d.quiver(dx, dy, dz, *direction, length=60, normalize=True,
                       color=colour, arrow_length_ratio=0.12)
    labels = {"family": "Times New Roman", "size": 14, "style": "italic"}
    axis_3d.text(dx + 65, dy, dz, "x", fontdict=labels)
    axis_3d.text(dx, dy + 65, dz, "y", fontdict=labels)
    axis_3d.text(dx, dy, dz + 65, "z", fontdict=labels)

    # Freeze the box once the layout is known: the arrows and the limits then stay put
    # while the skeleton moves inside them, the way they do in the reference.
    fig_3d.canvas.draw()
    frozen = (axis_3d.get_xlim(), axis_3d.get_ylim(), axis_3d.get_zlim())
    axis_3d.set_xlim(*frozen[0])
    axis_3d.set_ylim(*frozen[1])
    axis_3d.set_zlim(*frozen[2])
    axis_3d.set_autoscale_on(False)

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
                source_height = image.shape[0]
                cut = source_height - 580
                cropped = image[:cut, :, :]
                # the reference divides by the *uncropped* height, which squeezes the
                # tile; 1920/1092*450 = 791 px wide, not 1687
                width = int(image.shape[1] / source_height * 450)
                resized = cv2.resize(cropped, (width, 450))
                panel = np.full((450, 1640, 3), 255, np.uint8)
                x0 = (1640 - width) // 2
                panel[:, x0:x0 + width] = resized
                axis_video.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

            # --- 3DPose: only the landmarks move; the frame, the arrows and the
            # limits stay where drawfigure_init put them ---
            xyz = xyz_all[frame]
            for line, chain in zip(lines, spec["chains"]):
                idx = [index[p] for p in chain]
                line.set_data(xyz[idx, 0], xyz[idx, 1])
                line.set_3d_properties(xyz[idx, 2], "z")
            markers.set_data(xyz[:, 0], xyz[:, 1])
            markers.set_3d_properties(xyz[:, 2], "z")

            index_name = f"{frame // stride:04d}"
            fig_video.savefig(staging / f"{index_name}_v.png", facecolor="white")
            fig_3d.savefig(staging / f"{index_name}_3.png", facecolor="white")
            top = cv2.imread(str(staging / f"{index_name}_v.png"))
            bottom = cv2.imread(str(staging / f"{index_name}_3.png"))
            width = min(top.shape[1], bottom.shape[1])
            cv2.imwrite(str(staging / f"{index_name}.png"),
                        np.vstack([top[:, :width], bottom[:, :width]]))
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
        plt.close("all")


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
