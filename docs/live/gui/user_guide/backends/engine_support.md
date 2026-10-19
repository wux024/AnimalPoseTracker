# Inference engines

Six engines are supported. Only the one you deploy with has to be installed.

## Engine matrix

| Engine | Artifact | Install | Devices | Notes |
|---|---|---|---|---|
| ONNX Runtime | `.onnx` | `onnxruntime` or `onnxruntime-gpu` | CPU, CUDA GPU, some NPUs | widest device range |
| OpenVINO | `.xml` + `.bin` | `openvino` | Intel CPU, GPU, NPU | best on Intel hardware |
| TensorRT | `.engine` | TensorRT Python API | NVIDIA GPU | fastest on NVIDIA, export is device-specific |
| CoreML | `.mlpackage` | `coremltools` | Apple silicon, macOS | export needs a Mac |
| CANN | `.om` | Ascend toolkit | Huawei Ascend | vendor toolchain |
| OpenCV DNN | `.onnx`, `.pb` | nothing | CPU (GPU with a custom build) | always available, slowest |

## Choosing an engine

| Situation | Engine |
|---|---|
| unknown deployment machine | ONNX Runtime (CPU) |
| NVIDIA GPU, latency matters | TensorRT |
| Intel CPU/GPU/NPU | OpenVINO |
| Mac or iOS | CoreML |
| Ascend accelerator | CANN |
| nothing installed, just check the artifact | OpenCV DNN |

## How the engine is chosen

Two places:

- **`infer`** — the artifact's format implies the engine; `--engine` overrides it when a format
  has more than one runtime.
- **the inferencer GUI** — the engine list is filtered by the artifact's format and the devices
  the runtime reports.

## Two-stage artifacts

A SimCC artifact (AnimalViTPose) runs through the same engines, but needs a detector:

```bash
animalpose-cli infer --config infer.yaml --detector-weights detector.onnx
```

The detector is independent of the pose engine — an ONNX detector with a TensorRT pose model is
a normal combination.

## Where each engine is documented

- [ONNX](onnx.md)
- [OpenVINO](openvino.md)
- [TensorRT](tensorrt.md)
- [CoreML](coreml.md)

CANN and OpenCV DNN follow the same pattern; their setup lives with the vendor toolchain.
