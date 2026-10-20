# Tracking and filtering a recorded sequence

Take the video you predicted in the [quickstart](01_quickstart_cli.md), attach identities
across frames, and smooth each track. Tracking and filtering run on the prediction records,
so changing the tracker or the filter does not re-train or re-export anything.

## The six trackers

`--tracker` accepts six algorithms, all reading the same per-frame detections:

| Tracker | Behaviour | When it helps |
|---|---|---|
| `bytetrack` | IoU association with a motion model | the default; fast and predictable |
| `botsort` | IoU plus appearance features | animals that cross or stop |
| `ocsort` | observation-centric association | long occlusions |
| `deepocsort` | OCSort with appearance | crowded scenes |
| `fasttrack` | lightweight IoU association | many short tracks |
| `tracktrack` | tracking-through-time association | dense, continuous motion |

Behaviour flags are shared across trackers: `--track-track-buffer` (frames a lost track
survives), `--track-match-thresh` (association cost), `--track-new-track-thresh` (birth),
`--track-min-track-len` (pruning).

## Steps

### 1. Predict with tracking

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/camA.mp4 \
    --tracker bytetrack --output-dir results/predict/camA-tracked
```

Three files can come out of one run:

| File | Contents |
|---|---|
| `predictions.json` | per-frame, per-instance keypoints and scores |
| `tracked_predictions.json` | the same records with a `track_id` added |
| `predictions-filter.json` | the smoothed sequence (only with `--pose-filter`) |

Each video gets a fresh tracker instance, so track IDs are local to that video — ID 3 in one
file is not ID 3 in another.

### 2. Filter each track

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/camA.mp4 \
    --tracker bytetrack --pose-filter median --filter-window 7 \
    --output-dir results/predict/camA-filtered
```

`--pose-filter` is `median` (robust to single-frame outliers) or `savgol` (Savitzky-Golay,
smoother but can overshoot sharp turns). The filter runs per track, after tracking; a filter
without a tracker is refused, because smoothing across identities would be meaningless.
Filtering only smooths detections that exist — it does not invent poses for missed frames
(`--filter-max-gap` only lets the smoother bridge a short hole).

### 3. Read the record files

The three files share one schema: a list of `{"frame": N, "predictions": [...]}` entries,
each prediction carrying keypoints, a score, and — in the tracked file — a `track_id`. That
schema is the contract for everything downstream: the filter consumes it, the 3D stage
consumes it, and your own analysis can read it with `json.load` and nothing else.

### 4. Replay the sequence with another tracker

Because the records are the contract, you can re-associate a saved sequence without
re-running the network:

```python
import json
from animalposetracker.tracking import create_tracker
from animalposetracker.postprocessing.workflows import track_frame_predictions

records = json.loads(open("results/predict/camA/predictions.json").read())
tracker = create_tracker("ocsort")
tracked = [
    {"frame": entry["frame"],
     "predictions": track_frame_predictions(entry["predictions"], tracker, entry["frame"])}
    for entry in records
]
```

This is the same code path the CLI uses; only the tracker changes. Try `ocsort` when
`bytetrack` loses an animal across an occlusion, or `botsort` when two animals cross.

## Where to go next

- [Multi-camera tracking](../../main-workflows/multi-camera-tracking.md) — carrying these
  identities into the 3D stage.
- [Pose analysis workflow](../../recipes/post.md) — the same chain from Python.
