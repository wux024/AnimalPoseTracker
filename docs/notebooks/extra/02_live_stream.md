# Live inference on a camera or stream

Run the exported artifact from the [export walkthrough](01_export_matrix.md) on a live
source and keep the per-frame records. `infer` is a standalone command: it reads no project
configuration — the artifact, its sidecar, and one optional config file are all it sees.

## The infer config file

```bash
animalpose-cli infer --write-template infer.yaml
```

The template lists every key with its default, commented out. Uncomment what you change;
command-line flags win over the file. Two keys matter most:

```yaml
poser_weights: results/export/best.onnx   # the exported pose artifact
source: 0                                  # camera index, video file, or stream URL
```

## Camera vs video vs stream

`--source` accepts three kinds of input:

- a **camera index** (`0`, `1`, ...) — opens the local device;
- a **video file** — runs at read speed, useful for validating a deployment offline;
- a **stream URL** — RTSP or HTTP, for a remote camera.

Smoke-test any source without committing to a full run:

```bash
animalpose-cli infer --poser-weights results/export/best.onnx \
    --source 0 --max-frames 30 --records smoke.json
```

## Tracking and filtering live

```bash
animalpose-cli infer --poser-weights results/export/best.onnx --source 0 \
    --tracker bytetrack --records live.json
```

Tracking attaches a `track_id` per detection while the stream runs. Filtering is different:
it runs **after the stream ends**, smoothing each track over the recorded sequence:

```bash
animalpose-cli infer --poser-weights results/export/best.onnx --source 0 \
    --tracker bytetrack --pose-filter savgol --records live.json
```

`--pose-filter` without `--tracker` is refused — the smoother needs track IDs to know which
detections belong together.

## The records file

`--records` writes one JSON list: `[{"frame": N, "predictions": [...]}, ...]`, each
prediction carrying keypoints, a score, and a `track_id` when tracking is on. This is the
same record schema `predict` writes, so live captures feed the same filtering and 3D stages
as offline runs.

Rendering is independent of recording: `--output live.mp4` writes the rendered stream,
`--show` pops a window (press `q` to quit), and both work with or without `--records`.

## The two-stage stream

For a SimCC (top-down) artifact, attach a detector or the inferencer refuses the model:

```bash
animalpose-cli infer --poser-weights vitpose.onnx --detector-weights rtdetr.onnx \
    --source 0 --records live.json
```

The stream becomes detector → box → crop → ViT → points mapped back. The detector is any
NMS-free artifact that emits boxes; `--detector-box-format`, `--detector-coord-mode` and
`--detector-preprocess-mode` cover the two common conventions (the defaults match RT-DETR;
YOLO exports use `xyxy` + `pixels` + `letterbox`).

## Where to go next

- [Live inference overview](../../live/index.md) — engines, devices and the GUI front-ends.
- [Tracking and filtering](../main_demos/04_tracking_and_filtering.md) — the tracker and
  filter choices in detail.
