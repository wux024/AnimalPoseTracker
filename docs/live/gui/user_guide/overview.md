# Inferencer overview

The window has one job: take an artifact and a source, and keep them running while writing what
it sees.

## Select config and weights

The configuration file describes the classes and keypoints — either a project YAML
(`classes_name` / `keypoint_names` / `skeleton`) or a dataset YAML (`names` / `kpt_shape` /
`skeleton`). The counts are not labels only: they tell the decoder how the output tensor is laid
out, so a configuration that disagrees with the artifact decodes nonsense.

The weights are the exported artifact. The window reads its sidecar and reports a mismatch before
the run starts.

## Engine and device

The engine follows the artifact's format; the device list comes from the runtime. A CPU-only
runtime shows one device, a CUDA build shows the GPUs, OpenVINO shows CPU/GPU/NPU.

Precision is a property of the artifact: exporting FP32 and running with `--bits 8` is not a
conversion, and the window will not pretend it is.

## Start and tune

| Control | What it changes |
|---|---|
| confidence | detections below it are dropped |
| NMS IoU | overlap threshold for suppression |
| **Track** | adds a track id to every detection |
| **Filter** | smooths each track (`median` / `savgol`) |
| visualization toggles | boxes, keypoints, skeleton, scores |

Filtering needs track ids, so its controls stay disabled until **Track** is ticked — the same
dependency the CLI enforces between `--pose-filter` and `--tracker`.

## Outputs

| Path | Contents |
|---|---|
| `results/inference/records.json` | `[{frame, predictions}, ...]`, with `track_id` when tracking was on |
| `results/inference/output.avi` | the rendered video, when **Save Video** is used |

There is no switch for the records: they are the run's result. The same file can be replayed with
another tracker or filter later, because tracking and filtering read the records rather than the
pixels.

## Where to go next

- [Device and precision](device_config.md)
- [Inference engines](backends/engine_support.md)
- [Records format](misc/records_format.md)
