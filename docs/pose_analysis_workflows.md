# Pose analysis workflow

AnimalPoseTracker keeps prediction, tracking, temporal filtering, and 3D
reconstruction as separate API stages. A project can opt into the video stages
without changing the live inference engine or GUI.

## Video prediction, tracking, and filtering

`AnimalPoseTrackerProject.predict()` accepts an optional tracker and pose filter:

```python
project.predict(
    inference_source=["videos/camera_1.mp4"],
    tracker="bytetrack",
    pose_filter="median",
)
```

This returns the selected output directory and starts the project prediction
worker. After the worker finishes, the directory contains the original
per-frame detections in `predictions.json`, confirmed tracks with `track_id` in
`tracked_predictions.json`, and smoothed track coordinates in
`filtered_predictions.json`. The rendered video displays track IDs when
tracking is enabled. Each video gets a new tracker instance, so track IDs are
local to that video.

This video chain works directly with AnimalRTPose. AnimalViTPose video
prediction still needs instance boxes; Python callers can supply a
`DetectorBoxProvider` through `predict_cli.run(..., box_provider=...)`.

The same options can be kept in `configs/other.yaml`:

```yaml
tracking:
  enabled: true
  algorithm: bytetrack
  track_buffer: 30
  gmc_method: none

pose_filter:
  method: median
  window_length: 5
  polyorder: 2
  confidence_threshold: 0.25
```

If a pose filter is requested without an explicit tracker, the workflow uses
ByteTrack so that filtering is performed separately for each identity. Set
`gmc_method` to a supported camera-motion method only when video frames are
available; the project video workflow supplies those frames to the tracker.
Filtering smooths detections that exist in the tracked sequence. It does not
invent boxes or pose detections for missed frames.

## Synchronized multi-camera 3D

After obtaining one tracked prediction file per synchronized camera, load the
camera calibrations as `CameraCalibration` objects or mappings and pass the
files to `project.triangulate()`:

```python
from animalposetracker.pose3d import CameraCalibration

cameras = {
    "left": CameraCalibration.from_mapping(left_calibration),
    "right": CameraCalibration.from_mapping(right_calibration),
    "top": CameraCalibration.from_mapping(top_calibration),
}

results_3d = project.triangulate(
    predictions_by_camera={
        "left": "runs/predict/left-camera/tracked_predictions.json",
        "right": "runs/predict/right-camera/tracked_predictions.json",
        "top": "runs/predict/top-camera/tracked_predictions.json",
    },
    cameras=cameras,
    source_by_camera={
        "left": "videos/left-camera.mp4",
        "right": "videos/right-camera.mp4",
        "top": "videos/top-camera.mp4",
    },
)
```

Each file may contain several videos; `source_by_camera` selects the synchronized
video in each file. The API requires matching frame indices and camera IDs. If each camera has one
animal, it assigns the shared identity `animal_0`. For multiple animals, pass an
explicit `identity_map`, for example `{"left": {1: "mouse_a"}, "right":
{4: "mouse_a"}}`, to map camera-local track IDs to the same animal. The result
contains one `TriangulationSequence` per identity, including XYZ coordinates,
confidence, reprojection error, view counts, and inlier cameras.

This stage assumes camera synchronization, calibration, and cross-camera
identity correspondence are already available. It does not estimate those
inputs automatically.
