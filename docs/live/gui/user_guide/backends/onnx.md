# ONNX engine

Runs `.onnx` artifacts through the ONNX runtime.

## Artifact

| Property | Value |
|---|---|
| Format | `.onnx` |
| Install | `pip install onnxruntime (CPU) or onnxruntime-gpu` |
| Devices | CPU, CUDA GPU, some NPUs |

## When to use it

Pick this engine when the deployment machine is CPU and the
artifact was exported for it. The engine is implied by the artifact's extension;
`--engine` overrides the choice when a format has several runtimes.

## Limits

- providers: CPUExecutionProvider, CUDAExecutionProvider
- dynamic shapes follow the exported axes
- INT8 needs a quantised artifact, not just the runtime

## Checks

```bash
animalpose-cli infer --config infer.yaml
```

The first frames report the engine and device in use; a mismatch between the artifact
and the runtime fails there rather than mid-stream.
