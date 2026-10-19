# Multi-camera tracking and identity association

The 3D stage does not track by itself: it consumes the 2D records, associates one instance per
camera, and triangulates. This page covers the two places identity can be lost — across frames
and across cameras — and what the toolbox does about each.

## Pose records as the tracking input

Everything starts from `predictions.json`, one per camera:

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/camA.mp4 \
    --tracker bytetrack --pose-filter median --output-dir results/predict/camA
```

Three files can come out of one prediction run:

| File | Contents |
|---|---|
| `predictions.json` | per-frame, per-instance keypoints and scores |
| `tracked_predictions.json` | the same records with a track id added |
| `predictions-filter.json` | the smoothed sequence |

The 3D stage reads `predictions.json` by default; point `--views camera=path` at a tracked or
filtered file when you want those identities to carry into 3D.

## Trackers

Six trackers ship, all reading the same records:

| Tracker | Behaviour | When it helps |
|---|---|---|
| `bytetrack` | IoU association with a motion model | the default; fast and predictable |
| `botsort` | IoU plus appearance features | animals that cross or stop |
| `ocsort` | observation-centric association | long occlusions |
| `deepocsort` | OCSort with appearance | crowded scenes |
| `fasttrack` | lightweight IoU association | many short tracks |
| `tracktrack` | tracking-through-time association | dense, continuous motion |

The flags that change behaviour are the same across trackers:
`--track-track-buffer` (frames a lost track survives), `--track-match-thresh` (association
cost), `--track-new-track-thresh` (birth), `--track-min-track-len` (pruning).

## Association across cameras

Triangulation needs to know which detection in camera A is the same animal as which in camera
B. `pose3d --association` chooses how:

- **`index`** (default) pairs the n-th instance of every camera. It is only correct with one
  animal per frame, and it silently produces wrong 3D points when that assumption breaks.
- **`epipolar`** matches by the epipolar constraint: for a point in camera A, the candidate in
  camera B must lie near the corresponding epipolar line. Cost is the Sampson distance, capped
  by `--max-sampson` (default 25 px).

```bash
animalpose-cli pose3d --config configs/runtime.yaml --association epipolar
```

`epipolar` is what keeps identities stable when detection order changes, a detection drops in
one view, or two animals cross. It needs a calibrated rig; with a bad calibration the epipolar
lines point at the wrong place and every match fails.

## Identity in the output

Each instance in `poses3d.json` records `view_indices` — the instance index chosen in each
camera. That is the audit trail: if a 3D skeleton jumps, `view_indices` shows which camera's
detection changed.

`report.json` records the association statistics under `association_stats`: how many frames
were solved with how many views, and how many candidates were rejected. A sudden drop in the
average view count means the association started failing, not that the animals moved.

## Sync and failure modes

`--check-sync` compares the reprojection residual over time across views. A drift means the
cameras are not synchronized; a constant offset means one view is mis-calibrated.

| Symptom | Likely cause |
|---|---|
| 3D skeleton jitters along one axis | that camera's extrinsics are off |
| identity swaps when two animals cross | `index` association, or a weak tracker |
| average view count drops mid-sequence | a detection disappeared and `epipolar` rejected the wrong match |
| 3D points look mirrored | world-axis convention differs between the calibration and the renderer |

The overlay render is the fastest way to see which view is wrong: each 3D point is projected
back through the rig, so the camera where the skeleton slides off is the one to recalibrate.

## Putting it together

```bash
# 1. predict each camera into its own directory, with tracking and filtering
for cam in camA camB camC; do
  animalpose-cli predict --config configs/runtime.yaml \
      --weights results/train/weights/best.pt --source videos/$cam.mp4 \
      --tracker bytetrack --pose-filter median \
      --output-dir results/predict/$cam
done

# 2. associate and triangulate
animalpose-cli pose3d --config configs/runtime.yaml --association epipolar --check-sync

# 3. inspect
animalpose-cli pose3d-render --config configs/runtime.yaml --video overlay --images
```

Step 3 is the check that matters: if the skeleton lands on the animal in every view, the
records, the association and the calibration all agree.
