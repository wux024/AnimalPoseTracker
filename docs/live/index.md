# Live inference

`animalpose-cli infer` runs an exported artifact on a camera, a video file or a stream URL. It
reads no project configuration: its own YAML is the whole surface.

## What `infer` is

The deployment-side counterpart of `predict`:

| | `predict` | `infer` |
|---|---|---|
| input | a project and a checkpoint | an exported artifact |
| needs PyTorch | yes | no |
| needs a project | yes | no |
| sources | images, folders, splits | camera, video, stream |
| output | `predictions.json` + renders | `records.json` + renders |

`infer` refuses `.pt` checkpoints — export first. That keeps the deployment path free of the
training stack, which is what makes an artifact portable.

## Install the engine

Only the engine you deploy with has to be installed:

| Format | Install |
|---|---|
| ONNX | `onnxruntime` (CPU) or `onnxruntime-gpu` |
| OpenVINO | `openvino` |
| TensorRT | TensorRT Python API |
| CoreML | `coremltools` |
| CANN | Ascend toolkit |
| OpenCV | nothing |

The [installation page](../installation.md) has the versions and the extras.

## The config file

```bash
animalpose-cli infer --write-template              # writes ./infer.example.yaml
animalpose-cli infer --write-template -             # prints it
```

The template is the whole surface: poser weights, engine and device, source, thresholds,
drawing, tracking and filtering, plus the class/keypoint description (`names` / `kpt_shape` /
`skeleton`) that decodes the output tensor.

```bash
animalpose-cli infer --config infer.yaml
animalpose-cli infer --config infer.yaml --conf 0.4   # an explicit flag wins for that run
```

`--poser-weights` and `--source` have no defaults: the file or the command line must supply
them.

## Sources

- **Video file** — any format OpenCV decodes.
- **Camera** — an index; the device must support the requested resolution and FPS.
- **Stream URL** — RTSP or HTTP.

## Two-stage inference

A SimCC artifact (AnimalViTPose) needs boxes. Supply a detector:

```bash
animalpose-cli infer --config infer.yaml --detector-weights detector.onnx
```

Without the flag, the inferencer refuses the artifact and names the flag. With it, the stream
becomes: detector → box → crop → ViT → points mapped back to the frame.

The detector is any artifact that emits boxes; `DetectorBoxProvider` adapts it to the
`boxes_for_image` interface the pipeline expects.

## Records, tracking and filtering

Every run writes `results/inference/records.json`, in the `[{frame, predictions}, ...]` shape,
with a `track_id` on every detection when tracking was on.

- **tracking** — `--tracker` with the same six algorithms `predict` accepts;
- **filtering** — `--pose-filter median|savgol` with window, order and gap settings.

Filtering needs track ids, so it is rejected without a tracker — the same rule the GUI enforces
by disabling the controls.

## Engine matrix

| Engine | Artifact | Devices |
|---|---|---|
| ONNX Runtime | `.onnx` | CPU, CUDA GPU, some NPUs |
| OpenVINO | `.xml` + `.bin` | Intel CPU, GPU, NPU |
| TensorRT | `.engine` | NVIDIA GPU |
| CoreML | `.mlpackage` | Apple silicon, macOS |
| CANN | `.om` | Huawei Ascend |
| OpenCV DNN | `.onnx`, `.pb` | CPU, optionally GPU |

## Failure modes

| Symptom | Cause |
|---|---|
| `infer` rejects the weights | a `.pt` checkpoint — export it first |
| "does not match the configuration" | the sidecar and the YAML disagree on classes or keypoints |
| a SimCC artifact is refused | top-down needs `--detector-weights` |
| keypoints decode into nonsense | `kpt_shape` in the YAML disagrees with the artifact's output layout |
| the stream stalls on a camera | resolution or FPS the device does not support |
