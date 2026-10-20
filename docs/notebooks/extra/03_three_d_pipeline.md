# Full 3D pipeline

From a bare rig to a rendered 3D sequence: create a 3D project, calibrate with a ChArUco
board, triangulate the 2D records, and check the result visually. Two or more synchronized
cameras are required — one camera cannot produce a 3D point.

## Steps

### 1. Create the 3D project

```bash
animalpose-cli create --dataset trimouse --dataset-root /path/to/trimouse \
    --workspace projects --name demo3d --worker me \
    --model AnimalRTPose --scale N --date demo --pose3d
cd projects/demo3d-me-AnimalRTPose-N-demo
```

`--pose3d` adds `configs/cameras.yaml` (written by `calibrate`, read by `pose3d`) and
`results/pose3d/`. Public-benchmark descriptors are refused — they ship no multi-camera
capture, so there is nothing to calibrate against.

### 2. Calibrate the rig

Print a ChArUco board, measure the print (the square size sets the world scale), wave it
through the shared view of all cameras, and run:

```bash
animalpose-cli calibrate \
    --view camA=frames/camA --view camB=frames/camB --view camC=frames/camC \
    --board-squares 10x7 --square-size 22 --marker-size 16.5 \
    --output configs/cameras.yaml
```

One `--view` per camera, an image folder or a video each. The fit is a joint bundle
adjustment across all views; `--holdout` (default 0.2) keeps frames back so the reported
error is honest. Read that number before going further: a camera whose held-out
reprojection error is an order of magnitude above the others usually means a mis-detected
board or a wrong square size. Re-running on the same rig later can reuse the lens fit with
`--intrinsics-from configs/cameras.yaml`.

### 3. Predict one file per camera

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/camA.mp4 \
    --tracker bytetrack --output-dir results/predict/camA
```

Repeat per camera, each into its own `results/predict/<camera>/` directory — that layout is
what `pose3d` reads by default (`--views camera=path` overrides it). Tracking here is not
decoration: tracked identities are what cross-camera association can trust.

### 4. Triangulate

```bash
animalpose-cli pose3d --config configs/runtime.yaml --association epipolar
```

`--association index` (the default) pairs the n-th detection of every camera and is only
correct with exactly one animal per frame; `epipolar` matches by the epipolar constraint,
so a dropped detection or two animals crossing cannot swap identities. Frames are solved in
blocks of 2048 — one batched call per block, not one SVD per joint — with outlier cameras
dropped and re-solved per joint. Outputs land in `results/pose3d/`:

| File | Contents |
|---|---|
| `poses3d.json` | per-frame 3D keypoints with confidence, validity, reprojection error, contributing views |
| `report.json` | calibration snapshot, thresholds, sync check, association statistics |

### 5. The overlay check

```bash
animalpose-cli pose3d-render --config configs/runtime.yaml --video both
```

`--video overlay` projects the 3D keypoints back through the rig onto each camera's own
video — this is the decisive calibration check. A correct rig lands the skeleton on the
animal in every view; a wrong one slides off, and the direction it slides tells you which
camera is off. `--video space` gives the revolving 3D view; `--video both` writes both into
`results/pose3d/videos/`. Add `--plot results/pose3d/trajectory.png` for one instance's 3D
trajectory, or run bare for the flat CSV.

## Where to go next

- [3D AnimalPoseTracker](../../Overviewof3D.md) — the stage end to end with every option.
- [Multi-camera tracking](../../main-workflows/multi-camera-tracking.md) — the identity side
  of the pipeline.
