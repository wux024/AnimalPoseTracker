# Multi-camera 3D

Two or more synchronized cameras make a third dimension reachable: the same keypoint seen
by each view intersects at one world point. This page covers the stage end to end — fitting
the rig, triangulating the 2D records, and rendering the result.

A project carries the 3D stage only if it was created with `--pose3d`. The flag lives in
`project.yaml` and drives the extra directory and the GUI tab; public-benchmark projects are
refused because they ship no multi-camera capture.

## Fit the rig

`calibrate` takes one ChArUco view per camera — an image folder or a video — fits one
constrained lens model per camera, and writes `configs/cameras.yaml`:

```bash
animalpose-cli calibrate \
    --view camA=frames/camA --view camB=frames/camB --view camC=frames/camC \
    --board-squares 10x7 --square-size 22 --marker-size 16.5 \
    --output configs/cameras.yaml
```

| Option | Meaning |
|---|---|
| `--view camera=path` | Repeat once per camera; an image directory or a video |
| `--board-squares 10x7` | ChArUco squares, columns × rows |
| `--square-size` / `--marker-size` | Millimetres; they set the world scale, so measure the print |
| `--dictionary` | ArUco dictionary, default `DICT_4X4_50` |
| `--frame-step` / `--max-frames` | Video sampling |
| `--intrinsic-frames` | Frames used for the lens fit, default 60 — OpenCV slows down sharply past this |
| `--intrinsics-from` | Reuse the lens model and distortion of an existing calibration |
| `--sync hardware\|software\|unknown` | Recorded in the rig and reported by `pose3d --check-sync` |
| `--force` | Required to overwrite an existing `cameras.yaml` |

The fit uses a bundle adjustment over all views; `--skip-joint` falls back to the PnP median.
`--holdout` (default 0.2) keeps frames back for the reported error.

## Triangulate

`pose3d` reads one `predictions.json` per camera and writes into `results/pose3d/`:

| File | Contents |
|---|---|
| `poses3d.json` | Every frame with each instance's 3D keypoints, per-point confidence, validity, reprojection error and the contributing views |
| `poses3d-filter.json` | The temporally filtered sequence, when `--pose-filter` is used |
| `report.json` | Calibration snapshot, thresholds, sync check and association statistics |

```bash
animalpose-cli pose3d --config configs/runtime.yaml
animalpose-cli pose3d --config configs/runtime.yaml --association epipolar
animalpose-cli pose3d --config configs/runtime.yaml --check-sync
animalpose-cli pose3d --config configs/runtime.yaml --no-refine
animalpose-cli pose3d --config configs/runtime.yaml --backend numba
```

Every frame is solved in blocks: one batched call per block of 2048 frames instead of one
SVD per joint. `numpy` is the default and needs nothing extra; `numba` and `jax-gpu` need
their extras and only pay off past roughly 60k points per block. There is no per-joint
RANSAC — the whole camera set is solved first, a view that reprojects past
`--reprojection-threshold` is dropped, and the joint is solved again from the rest.
`--no-refine` skips the per-joint least-squares polish. On 200 frames of 17 keypoints a
block takes 4.6 s, or 0.9 s with `--no-refine`; `report.json` records the settings under
`triangulation`.

`--views camera=path` overrides the per-camera prediction file; by default each camera is
read from `results/predict/<camera>/predictions.json`, which is why multi-camera prediction
should write each camera into its own directory:

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/camA.mp4 \
    --tracker bytetrack --pose-filter median --output-dir results/predict/camA
```

### Associating instances across cameras

`--association` decides which detection in one camera belongs to which in another:

- `index` (default) pairs the n-th instance of every camera. It is only correct with one
  animal per frame.
- `epipolar` matches by the epipolar constraint, so detection order, a dropped detection,
  or two animals crossing cannot swap identities. Cost is the Sampson distance, capped by
  `--max-sampson` (default 25 px).

Each instance in `poses3d.json` records the chosen instance per camera (`view_indices`);
`report.json` carries the match counts under `association_stats`.

## Render

`pose3d-render` turns the sequence into files for reading, plotting and watching:

```bash
animalpose-cli pose3d-render --config configs/runtime.yaml
animalpose-cli pose3d-render --config configs/runtime.yaml --plot results/pose3d/trajectory.png
animalpose-cli pose3d-render --config configs/runtime.yaml --video both
animalpose-cli pose3d-render --config configs/runtime.yaml --filtered --video overlay --images
```

| Output | Option | Path |
|---|---|---|
| CSV, one row per (frame, instance, keypoint) | default | `results/pose3d/poses3d.csv` |
| JSON summary | `--summary-output` | next to the CSV |
| Trajectory PNG for one instance | `--plot` | given path |
| Reprojection video, cameras tiled | `--video overlay` | `results/pose3d/videos/poses3d-overlay.mp4` |
| Revolving 3D view | `--video space` | `results/pose3d/videos/poses3d-space.mp4` |
| Both | `--video both` | both files |
| One JPEG per frame | `--images` | `results/pose3d/images/` |

`--filtered` reads `poses3d-filter.json` and writes into `videos-filter/` and
`images-filter/`. `--layout vertical` stacks the cameras instead of placing them side by
side; `--stride` keeps every n-th frame; `--fps` sets the output rate.

The overlay is the practical check on a calibration: each 3D keypoint is projected through
the rig onto the frame that recorded it, so a correct rig lands the skeleton on the animal
and a wrong one slides off. It needs the 2D predictions to come from videos (the command
reads the source video named in each `predictions.json`); sequences predicted from image
folders can still be rendered with `--video space`.

## The same stage in the GUI

The **3D Reconstruction** tab appears only for a project created with the 3D option. It
lists the cameras, lets you double-click a cell to pick that camera's calibration folder or
video and its predictions file, and runs `calibrate`, `pose3d` and `pose3d-render` with the
live output in the page. The commands it runs are the CLI's own; the page only collects the
paths and numbers.

A project created without 3D can gain it later from **Tools → Enable 3D reconstruction for
this project**, which writes the flag, the empty `configs/cameras.yaml` and
`results/pose3d` without touching anything else. Public-benchmark projects stay locked out.
