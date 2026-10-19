# Inferencer GUI

`animalpose-inferencer` runs an exported artifact — no project, no checkpoint — on a camera, a
video file or a stream URL. It is the deployment window: what it loads is what you ship.

## What the window is for

- Read an ONNX, OpenVINO, TensorRT, CoreML, CANN or OpenCV artifact.
- Preview a camera or a video before starting.
- Draw, track and filter the poses live.
- Write the run's records and a rendered video.

It does not train or evaluate; the workspace does that.

## Launch and load

```bash
animalpose-inferencer
```

Pick the configuration file and the poser weights. The available engines and devices appear
automatically, filtered by what the artifact can run on.

https://github.com/user-attachments/assets/db726dd8-a092-40e3-94ea-e3d423973a03

The configuration file describes the classes and keypoints. Either style is accepted: a
project YAML (`classes_name` / `keypoints_name` / `skeleton`) or a dataset YAML (`names` /
`kpt_shape` / `skeleton`). These keys are not only labels — their counts tell the decoder how
the output tensor is laid out, so a configuration that disagrees with the artifact shifts the
keypoint block and decodes nonsense.

## Choosing a source

- **Video file** — any format OpenCV can decode.
- **Camera** — pick the index; set resolution and FPS before starting, and make sure the
  camera supports them.
- **Stream URL** — an RTSP or HTTP stream.

https://github.com/user-attachments/assets/007caeb6-b5b2-48a8-9873-58ff9db002b7

## Engine and device

The engine is implied by the artifact format; the device list comes from the runtime. A CPU
device always works, a GPU device needs the matching build (CUDA for ONNX Runtime GPU and
TensorRT, the Intel runtime for OpenVINO, and so on).

The full matrix, with the extra each engine needs, is in
[Installing AnimalPoseTracker](../../installation.md).

## Starting a run

Set the thresholds, then start:

| Control | Effect |
|---|---|
| confidence | detections below it are dropped |
| NMS IoU | overlap threshold for suppression |
| **Track** | adds a track id to every detection |
| **Filter** | smooths each track (`median` or `savgol`) |
| visualization toggles | boxes, keypoints, skeleton, scores |

Filtering needs track ids, so the filter controls stay disabled until **Track** is ticked —
the same rule `--pose-filter` follows on the command line.

https://github.com/user-attachments/assets/0b57bba6-9ed6-47b1-9f4f-1d449762c784

## Records and output

Every run writes `results/inference/records.json`, in the same
`[{frame, predictions}, ...]` shape `infer --records` writes, with a `track_id` on every
detection when tracking was on. There is no switch for it: the records are the run's result.

`Save Video` writes the rendered frames to `results/inference/output.avi`.

## Two-stage artifacts

A SimCC artifact (AnimalViTPose) is refused at selection — this window infers one-stage
layouts only. Run those through `animalposetracker predict`, or use
`animalpose-cli infer --detector-weights` from the terminal.
