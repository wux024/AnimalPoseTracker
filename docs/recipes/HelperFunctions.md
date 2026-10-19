# Public API map

Where each capability lives, for people driving the toolbox from Python rather than the CLI.

## Project API

`animalposetracker.Project` is the entry point for a project:

| Call | Does |
|---|---|
| `Project.open(path)` | open an existing project directory |
| `Project.create(...)` | the same as `animalpose-cli create` |
| `.train(...)` | training, with the same flags the CLI exposes |
| `.evaluate(...)` | validation on the configured split |
| `.predict(...)` | prediction on a split, folder or video |
| `.export(format=...)` | export to a deployment format |

The object is a thin wrapper over the CLI paths: it resolves the config, calls the same
functions, and returns the same result dictionaries.

## Tracking API

`animalposetracker.tracking`:

| Function | Does |
|---|---|
| `track_records(records, tracker=...)` | adds ids to a pose record list |
| `available_trackers()` | the six names the CLI accepts |

The input is the `predictions.json` shape; the output is the same shape plus `track_id`.

## Post-processing API

`animalposetracker.postprocessing`:

| Function | Does |
|---|---|
| `filter_records(records, method="median", ...)` | smooths per track |
| `decode_outputs(...)` | the shared decoder used by training, validation and deployment |

`decode_outputs` is the contract that keeps a metric honest: if you reimplement decoding
elsewhere, the number you report no longer describes what ships.

## Pose3d API

`animalposetracker.pose3d`:

| Function | Does |
|---|---|
| `calibrate_rig(...)` | fit intrinsics and extrinsics from ChArUco views |
| `associate_views(...)` | match instances across cameras (index or epipolar) |
| `triangulate_groups(groups, cameras, ...)` | batched triangulation of associated poses |
| `reconstruct_project(project)` | the full stage: predict records → 3D → report |
| `render_project(project, ...)` | CSV, summary, plot, videos |

`triangulate_groups` is the same function the `pose3d` command calls; passing `backend="numpy"`
or `"numba"` selects the batched solver.

## Inference API

`animalposetracker.inference`:

| Function | Does |
|---|---|
| `InferenceEngine(config=..., weights_path=...)` | loads an exported artifact |
| `.model_init()` | resolves the runtime and the output shapes |
| `resolve_output_schema(...)` | decides which decoder the artifact needs |

The inferencer has no dependency on the training stack: an artifact plus this module runs on a
machine with no PyTorch.

## Export API

`animalposetracker.export`:

| Function | Does |
|---|---|
| `export_model(weights, format=..., ...)` | one export call, same as the CLI |
| `available_formats()` | the rows of the [deployment matrix](OpenVINO.md) |

Each format is loaded lazily, so the import cost is paid only when that format is used.
