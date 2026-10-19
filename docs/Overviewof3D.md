# 3D AnimalPoseTracker

Two or more synchronized cameras make a third dimension reachable: the same keypoint seen
by each view intersects at one world point. This page covers the stage end to end — creating
a 3D project, calibrating the rig, triangulating the 2D records, and rendering the result.

## **ATTENTION: this stage assumes you:**

- **At least two synchronized cameras.** The stage triangulates across views; one camera
  cannot produce a 3D point.
- **A project created with `--pose3d`.** The flag lives in `project.yaml` and drives the
  extra configuration file, the results directory and the GUI tab.
- **A ChArUco board you can print.** The square and marker sizes set the world scale, so
  measure the print rather than trusting the nominal size.
- **One prediction file per camera.** The triangulator reads
  `results/predict/<camera>/predictions.json` by default.

Public-benchmark projects are refused: they ship no multi-camera capture, so there is nothing
to calibrate against.

### **If you need more than two cameras**

Nothing in the stage is limited to two. `calibrate` takes one `--view` per camera and fits a
joint bundle adjustment across all of them; the association and the triangulation scale the
same way. More cameras buy robustness — a joint seen by four views survives one bad view.

## Jump in with direct multi-camera support:

The five steps below produce a 3D sequence from a calibrated rig and the 2D predictions you
already have.

### (1) Create a New 3D Project

Pass `--pose3d` at creation time:

```bash
animalpose-cli create --dataset trimouse --workspace projects --name demo3d --worker me \
    --model AnimalRTPose --scale N --date demo --pose3d
```

The project gains one extra file and one extra directory:

```text
projects/demo3d-me-AnimalRTPose-N-demo/
├── configs/
│   └── cameras.yaml        written by `calibrate`, read by `pose3d`
└── results/
    └── pose3d/             poses3d.json, report.json, CSV, videos/
```

An existing project can gain the stage later from **Tools → Enable 3D reconstruction for this
project**, which writes the flag, an empty `configs/cameras.yaml` and `results/pose3d/`
without touching anything else. The flag is one-way: public-benchmark projects stay locked
out.

### (2) ChArUco calibration

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

Shoot the board from the angles the animals will occupy: a board that only ever appears
flat-on gives a well-conditioned intrinsic fit and a poorly constrained extrinsic one.

### (3) Check calibration

Two checks catch most mistakes before triangulation:

**Reprojection error of the held-out frames.** `calibrate` prints it and records it in
`configs/cameras.yaml`; a per-camera error an order of magnitude above the others usually
means a mis-detected board or a wrong square size.

```bash
animalpose-cli calibrate --view camA=frames/camA --view camB=frames/camB \
    --board-squares 10x7 --square-size 22 --marker-size 16.5 \
    --output configs/cameras.yaml --force
```

**Temporal alignment.** `pose3d --check-sync` compares the reprojection residual over time
across views; a drift means the cameras are not synchronized.

```bash
animalpose-cli pose3d --config configs/runtime.yaml --check-sync
```

The decisive check is visual and comes after triangulation: render the overlay in step (5). A
correct rig lands the skeleton on the animal in every view; a wrong one slides off, and the
direction it slides tells you which camera is off.

### (4) Triangulate

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

#### Associating instances across cameras

`--association` decides which detection in one camera belongs to which in another:

- `index` (default) pairs the n-th instance of every camera. It is only correct with one
  animal per frame.
- `epipolar` matches by the epipolar constraint, so detection order, a dropped detection,
  or two animals crossing cannot swap identities. Cost is the Sampson distance, capped by
  `--max-sampson` (default 25 px).

Each instance in `poses3d.json` records the chosen instance per camera (`view_indices`);
`report.json` carries the match counts under `association_stats`.

### (5) Visualize

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

#### The same stage in the GUI

The **3D Reconstruction** tab appears only for a project created with the 3D option. It
lists the cameras, lets you double-click a cell to pick that camera's calibration folder or
video and its predictions file, and runs `calibrate`, `pose3d` and `pose3d-render` with the
live output in the page. The commands it runs are the CLI's own; the page only collects the
paths and numbers.

### If you use this code

Cite the toolbox and the paper the 3D formulation comes from — [Wu et al. 2022](https://doi.org/10.1016/j.engappai.2022.104667)
for the motion-parameter formulation and [Wu et al. 2024](https://doi.org/10.1109/JSEN.2024.3374236)
for the tracked-keypoint perception system. See [Citation](citation.md).
