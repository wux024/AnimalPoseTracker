# Installing the inferencer

The inferencer is part of the package; what you install on top is the engine runtime for the
artifact you ship.

## Extras

```bash
pip install -v -e ".[gui]"              # the window
pip install onnxruntime                 # ONNX, CPU
pip install onnxruntime-gpu             # ONNX, CUDA (uninstall the CPU wheel first)
pip install openvino                    # Intel CPU/GPU/NPU
```

TensorRT, CoreML and CANN come from their vendor toolchains rather than from a Python extra; the
[engine matrix](../user_guide/backends/engine_support.md) lists what each needs.

## Artifact preparation

The inferencer reads exported artifacts only:

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx --dynamic --opset 17
```

Export writes the artifact and its `.json` sidecar. Keep them together: the sidecar carries the
preprocessing and the class/keypoint contract, and the window refuses an artifact whose sidecar
disagrees with the configuration.

## Launch

```bash
animalpose-inferencer
```

Pick the configuration file and the poser weights; the engine and device lists fill in from what
the runtime reports. If the weights are a SimCC artifact, the window refuses it — two-stage
inference belongs to `animalpose-cli infer --detector-weights`.
