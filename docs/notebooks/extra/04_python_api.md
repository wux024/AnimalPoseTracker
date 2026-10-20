# Driving the pipeline from Python

Everything the CLI does is a thin layer over a small API. This walkthrough drives one
project end to end — open it, train, predict with tracking, filter, triangulate, export —
without touching the command line.

## Opening a project

```python
from animalposetracker.project.api import AnimalPoseTrackerProject

project = AnimalPoseTrackerProject()
project.load_project_config("projects/demo-me-AnimalRTPose-N-demo/project.yaml")
```

`load_project_config` reads the project file and its three `configs/` companions; the object
now carries the dataset description, the model family and the runtime settings.

## Training and validating

```python
project.train()      # runs the configured training; writes results/train/
project.evaluate()   # validates the best checkpoint; writes results/val/metrics.json
```

Both are blocking calls over the same engine the CLI drives. `project.resume()` continues
from `results/train/weights/last.pt`, or from a path you pass.

## Prediction with tracking and filtering

```python
output_dir = project.predict(
    inference_source=["videos/camA.mp4"],
    tracker="bytetrack",
    pose_filter="median",
)
```

The call returns the run directory. It holds `predictions.json` (per-frame detections),
`tracked_predictions.json` (with `track_id`) and `predictions-filter.json` (smoothed) — the
same three files the CLI writes, with the same schema. Each video gets a fresh tracker, so
track IDs are local to a video.

The same records can be re-associated or re-smoothed without touching the network:

```python
import json
from animalposetracker.tracking import create_tracker
from animalposetracker.postprocessing.workflows import (
    track_frame_predictions, filter_tracked_video_records,
)

records = json.loads(open(output_dir / "predictions.json").read())
tracker = create_tracker({"algorithm": "ocsort", "track_buffer": 45})
tracked = [
    {"frame": entry["frame"],
     "predictions": track_frame_predictions(entry["predictions"], tracker, entry["frame"])}
    for entry in records
]
smoothed = filter_tracked_video_records(tracked, method="savgol", window_length=7)
```

`track_frame_predictions` is the exact function the CLI feeds one frame at a time;
`filter_tracked_video_records` smooths each track independently and never synthesizes
detections for missed frames.

## Multi-camera triangulation

With one tracked prediction file per synchronized camera:

```python
from animalposetracker.pose3d import CameraCalibration

cameras = {
    "left": CameraCalibration.from_mapping(left_calibration),
    "right": CameraCalibration.from_mapping(right_calibration),
}

result = project.triangulate(
    predictions_by_camera={
        "left": "results/predict/left/tracked_predictions.json",
        "right": "results/predict/right/tracked_predictions.json",
    },
    cameras=cameras,
    identity_map={"left": {1: "mouse_a"}, "right": {4: "mouse_a"}},
)
```

`identity_map` translates camera-local track IDs to shared animal names — with one animal
per camera it can be omitted. The result carries one triangulated sequence per identity:
XYZ, confidence, reprojection error, view counts. Calibration, synchronization and the
identity mapping are inputs to this stage; it does not estimate them.

## Exporting

```python
project.export(format="onnx")
```

Same exporter as the CLI: the artifact and its sidecar land under `results/export/`, and the
sidecar is what makes the artifact runnable by `infer` without a project.

## Where to go next

- [Pose analysis workflow](../../recipes/post.md) — the prediction/tracking/3D API with the
  configuration-level options.
- [3D AnimalPoseTracker](../../Overviewof3D.md) — what the triangulation stage assumes.
