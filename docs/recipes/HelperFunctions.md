# Public API map

Where each capability lives, for people driving the toolbox from Python rather than the CLI.
Every name below exists in the installed package; the signatures are the ones the CLI calls.

## Project API

`animalposetracker.project.api.AnimalPoseTrackerProject` is the entry point:

```python
from animalposetracker.project.api import AnimalPoseTrackerProject

project = AnimalPoseTrackerProject()
project.load_project_config("projects/demo-me-AnimalRTPose-N-demo/project.yaml")
```

| Call | Does |
|---|---|
| `create_new_project()` | the same as `animalpose-cli create` |
| `create_public_dataset_project(name)` | create from a packaged benchmark descriptor |
| `load_project_config(path)` | read `project.yaml` and its `configs/` companions |
| `train()` | one training run, configured by `configs/runtime.yaml` |
| `evaluate()` | validation on the configured split, writes `results/val/metrics.json` |
| `predict(inference_source=None, split=None, tracker=None, pose_filter=None)` | prediction; returns the run directory |
| `resume(resume_path)` | continue from `last.pt` or a path you pass |
| `export(format=None)` | export the current checkpoint |
| `triangulate(predictions_by_camera, cameras, identity_map=None, source_by_camera=None)` | the 3D stage |
| `enable_pose3d()` | add the 3D flag and `configs/cameras.yaml` to an existing project |
| `get_runtime_value(...)` / `set_runtime_value(...)` | read or set one runtime key |
| `request_training_stop()` | ask a running training to stop at the next batch |

`train()`, `evaluate()` and `export()` take no per-call overrides — set values in
`configs/runtime.yaml` (or use the CLI) first.

## Tracking API

`animalposetracker.tracking`:

| Name | Does |
|---|---|
| `create_tracker(config=None, **overrides)` | build one of the six trackers |
| `ALGORITHMS` | the names `--tracker` accepts |
| `TrackerConfig` | the per-tracker settings (`track_buffer`, `match_thresh`, …) |

`create_tracker("bytetrack")` and `create_tracker({"algorithm": "ocsort", "track_buffer": 45})`
both work; keyword overrides win over the mapping.

## Post-processing API

`animalposetracker.postprocessing`:

| Name | Does |
|---|---|
| `track_frame_predictions(detections, tracker, frame_index, frame=None, class_names=None)` | one frame through a tracker, returns detections with `track_id` |
| `filter_tracked_video_records(frame_records, method="median", window_length=5, polyorder=2, confidence_threshold=None, max_gap=0)` | smooth each track over a recorded sequence |
| `filter_pose_2d(...)` / `filter_pose_3d(...)` | the same smoothers for plain 2D/3D arrays |
| `decode_pose_outputs(...)` | the decoder training, validation and deployment share |

`decode_pose_outputs` is the contract that keeps a metric honest: a number produced by a
separate decoder no longer describes what ships.

## Pose3d API

`animalposetracker.pose3d`:

| Name | Does |
|---|---|
| `CameraCalibration` / `CameraRig` | one camera, or the whole rig; `CameraCalibration.from_mapping(...)` reads a calibration mapping |
| `load_camera_rig(path)` | read `configs/cameras.yaml` |
| `associate_instances(...)` | match instances across cameras (index or epipolar) |
| `triangulate_keypoints(...)` | one frame, one keypoint set |
| `triangulate_groups(groups, cameras, ...)` | batched triangulation, the function `pose3d` calls |
| `triangulate_sequence(...)` | a whole sequence |
| `reconstruct_project(project)` | predict records → 3D → report |

## Inference API

`animalposetracker.inference`:

| Name | Does |
|---|---|
| `InferenceEngine(config=..., weights_path=...)` | loads an exported artifact and its sidecar |
| `StreamInferencePipeline` | the camera/video loop behind `infer` and the GUI |
| `DetectorProvider` / `DetectorBox` | adapts a detector artifact to boxes for SimCC models |

The inferencer has no dependency on the training stack: an artifact plus this module runs on a
machine with no PyTorch.

## Export API

`animalposetracker.export`:

| Name | Does |
|---|---|
| `export_with_backend(weights, format=..., ...)` | one export call, the one the CLI makes |

Formats load lazily, so the import cost is paid only for the format you use. The format list is
the [deployment matrix](OpenVINO.md).
