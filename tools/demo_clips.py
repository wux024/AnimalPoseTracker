"""Render the demo clips from AdamPosePlug's data.

Two panels, nothing else: the video tile on top, the 3D skeleton underneath. Both follow
AdamPosePlugMain directly —

  * ``videoframe``: drop 580 px off the bottom, resize to 450 tall with the width taken
    from the *uncropped* height;
  * ``drawfigure_init``: no axes, blue skeleton, marker per keypoint, three coloured
    arrows anchored at (-150, -80, -200), z negated.

    python tools/demo_clips.py hand koi
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
}


def load_frames(path: Path, names: list[str]) -> np.ndarray:
    series = {name: [] for name in names}
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            for name in names:
                series[name].append([float(row[f"{name}_x"]), float(row[f"{name}_y"]),
                                     float(row[f"{name}_z"])])
    return np.stack([np.array(series[name]) for name in names], axis=1)


def render(name: str, spec: dict, limit: int | None, stride: int, fps: int) -> None:
    data = load_frames(SOURCE / spec["csv"], spec["points"])
    cap = cv2.VideoCapture(str(SOURCE / spec["video"]))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / height * 450)
    if frames != len(data):
        raise ValueError(f"{name}: {len(data)} rows against {frames} video frames")

    total = min(len(data), limit) if limit else len(data)
    index = {point: i for i, point in enumerate(spec["points"])}
    dx, dy, dz = -150.0, -80.0, -200.0
    colours = [tuple(c) for c in plt.cm.tab20(np.linspace(0, 1, len(spec["points"])))]

    # the canvas is as wide as the video tile, so neither panel has dead margins
    figure = plt.figure(figsize=(width / 100, 9.0), dpi=100)
    top = figure.add_axes([0.0, 0.5, 1.0, 0.5])
    bottom = figure.add_axes([0.0, 0.0, 1.0, 0.5], projection="3d")
    top.set_axis_off()

    # drawfigure_init(): everything below is drawn once. The arrows are anchored at
    # (-150, -80, -200) and never move; only the markers are updated per frame, which is
    # what keeps the axes still.
    bottom.set_axis_off()
    bottom.view_init(elev=30, azim=-60)      # the matplotlib default the reference uses

    # The box follows the data, not the arrows. The arrows are anchored at the corner of
    # that box, because keeping them at the literal (-150, -80, -200) stretched the z
    # axis to six hundred millimetres and shrank the skeleton to a knot in the middle.
    xyz_all = data.copy()
    xyz_all[:, :, 2] = -xyz_all[:, :, 2]
    low = xyz_all.min(axis=(0, 1))
    high = xyz_all.max(axis=(0, 1))
    pad = (high - low) * 0.08 + 1e-6
    low, high = low - pad, high + pad
    bottom.set_xlim(low[0], high[0])
    bottom.set_ylim(low[1], high[1])
    bottom.set_zlim(low[2], high[2])

    # arrows sit at the low corner, sized to the box so they keep the original proportion
    dx, dy, dz = float(low[0]), float(low[1]), float(low[2])
    arrow_len = float((high - low).max()) * 0.18
    label_offset = arrow_len * 1.1         # the reference puts the label just past the tip

    first = data[0].copy()
    first[:, 2] = -first[:, 2]
    skeletons = []
    for chain in spec["chains"]:
        idx = [index[p] for p in chain]
        skeletons.append(bottom.plot(first[idx, 0], first[idx, 1], first[idx, 2], "b-")[0])
    dots = [bottom.plot(first[i, 0], first[i, 1], first[i, 2], marker="o",
                        linestyle="none", color=colours[i], markersize=6)[0]
            for i in range(len(spec["points"]))]
    for direction, colour in (((1, 0, 0), "r"), ((0, 1, 0), "g"), ((0, 0, 1), "y")):
        bottom.quiver(dx, dy, dz, *direction, length=arrow_len, normalize=True,
                      color=colour, arrow_length_ratio=0.12)
    labels = {"family": "Times New Roman", "size": 14, "style": "italic"}
    bottom.text(dx + label_offset, dy, dz, "x", fontdict=labels)
    bottom.text(dx, dy + label_offset, dz, "y", fontdict=labels)
    bottom.text(dx, dy, dz + label_offset, "z", fontdict=labels)

    staging = Path(tempfile.mkdtemp(prefix=f"apt-{name}-"))
    try:
        for frame in range(total):
            ok, image = cap.read()
            if frame % stride:
                continue

            # videoframe(): 1092 - 580 px kept, width from the uncropped height
            top.clear()
            top.set_axis_off()
            if ok:
                image = image[: int(height - 580), :, :]
                top.imshow(cv2.cvtColor(cv2.resize(image, (width, 450)), cv2.COLOR_BGR2RGB))

            # drawfigure(): move the markers only, never redraw the axes
            xyz = data[frame].copy()
            xyz[:, 2] = -xyz[:, 2]
            for line, chain in zip(skeletons, spec["chains"]):
                idx = [index[p] for p in chain]
                line.set_data(xyz[idx, 0], xyz[idx, 1])
                line.set_3d_properties(xyz[idx, 2], "z")
            for i, dot in enumerate(dots):
                dot.set_data([xyz[i, 0]], [xyz[i, 1]])
                dot.set_3d_properties([xyz[i, 2]], "z")

            figure.savefig(staging / f"{frame // stride:04d}.png", facecolor="white")
        cap.release()

        OUTPUT.mkdir(parents=True, exist_ok=True)
        gif = OUTPUT / f"{name}.gif"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(staging / "%04d.png"), "-vf",
             f"fps={fps},scale=760:-1:flags=lanczos,split[a][b];"
             "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer", str(gif)],
            check=True,
        )
        print(f"{name}: {gif.stat().st_size // 1024} KB")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=list(SPECS) + [[]], default=list(SPECS))
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    for name in args.names or list(SPECS):
        render(name, SPECS[name], args.frames, max(1, args.stride), args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
