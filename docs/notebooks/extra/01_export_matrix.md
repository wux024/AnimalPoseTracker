# Export matrix in practice

One checkpoint, several deployment formats, and the sidecar file that makes each artifact
self-describing. Start from the trained model in the
[quickstart](../main_demos/01_quickstart_cli.md) (`results/train/weights/best.pt`).

## Steps

### 1. ONNX first

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx --opset 17
```

ONNX is the hub format: OpenVINO, TensorFlow, PaddlePaddle, MNN and NCNN exports all start
from this graph, so exporting it first doubles as a sanity check. The artifact lands in
`results/export/` next to a sidecar JSON.

### 2. The sidecar contract

Every export writes `<artifact>.json` beside the artifact. The inferencer reads it to learn
what the graph cannot say about itself:

| Key | Meaning |
|---|---|
| `head` | which decoder applies (`YOLOPoseHead` / `SimCCHead`) |
| `input_size`, `input_layout`, `input_dtype` | the tensor shape and type to feed |
| `color_order` | RGB — the graph does not carry this |
| `kpt_shape` | keypoint count and dims, for slicing the output |
| `outputs` | output tensor names (`predictions`, or `simcc_x` / `simcc_y`) |
| `pixel_mean` / `pixel_std` / `bbox_padding` | top-down preprocessing (SimCC heads only) |
| `simcc_split_ratio` | SimCC binning granularity (SimCC heads only) |
| `quantization` | how the artifact was quantized (`int8_static_qdq` / `int8_nncf` / `int8_tensorrt`; INT8 exports only) |
| `calibration_images` | how many images the calibration saw (INT8 exports only) |

Never edit or delete the sidecar for a SimCC artifact: mmpose-style graphs do not bake
normalization into the graph, so a missing sidecar means the inferencer feeds raw 0–255
pixels and the decoded pose scatters. A one-stage artifact can survive a missing sidecar if
you pass `--imgsz`; a top-down one cannot — without `pixel_mean`, `pixel_std`, `bbox_padding`
and `simcc_split_ratio` the decoder cannot reproduce the training preprocessing, so re-export
instead of running it.

### 3. The derivatives

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format openvino
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format engine --device 0
```

Each format needs its own SDK, loaded lazily — you only install what you export
(`.[export-openvino]`, `.[export-tensorflow]`, and so on; see
[Deployment formats](../../recipes/DeploymentFormats.md) for the per-format requirements). TensorRT
(`engine`) additionally requires a CUDA device at export time. Formats that need vendor
toolchains beyond pip (Edge TPU, IMX500, RKNN) fail with a format-specific message when the
pieces are missing — that is a dependency statement, not a crash.

### 4. Verify by predicting with the artifact

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/export/best.onnx --split val
```

`--weights` accepts an exported artifact directly; the sidecar tells the worker how to
preprocess and decode. Comparing this run's `predictions.json` against the `.pt` run is the
real export test — a format conversion that changes keypoints shows up here, not in the
exporter's exit code.

### 5. Precision and graph options

`--half` (FP16) and `--int8` shrink and speed up the artifact where the format supports it;
unsupported precision requests are rejected rather than silently ignored. `--int8` covers
onnx (static QDQ), openvino (nncf), TensorRT (entropy calibrator, cache kept beside the
engine), TFLite, Edge TPU, MNN and IMX500. The two never combine — `--half` with `--int8`
is rejected. Calibration reads the first 200 training images and preprocesses them exactly
as inference does; for onnx the FP32 graph is an intermediate and only the `_int8` artifact
is kept. `--dynamic` requests dynamic input axes for families whose export graph allows it;
`--simplify` runs the ONNX graph optimizer. The defaults are the safe path — change one flag
at a time and re-run step 4.

## Where to go next

- [Deployment formats](../../recipes/DeploymentFormats.md) — the full matrix with runtime
  requirements.
- [Live inference](02_live_stream.md) — running the artifact you just verified.
