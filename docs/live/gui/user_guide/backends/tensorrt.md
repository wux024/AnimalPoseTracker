# TensorRT engine

Runs `.engine` artifacts through the TensorRT runtime.

## Artifact

| Property | Value |
|---|---|
| Format | `.engine` |
| Install | `pip install TensorRT Python API + CUDA` |
| Devices | NVIDIA GPU |

## When to use it

Pick this engine when the deployment machine is nvidia gpu and the
artifact was exported for it. The engine is implied by the artifact's extension;
`--engine` overrides the choice when a format has several runtimes.

## Limits

- the engine is built for one GPU architecture
- building is slow; the artifact is not portable across cards
- FP16 is the usual choice; INT8 needs calibration

## Checks

```bash
animalpose-cli infer --config infer.yaml
```

The first frames report the engine and device in use; a mismatch between the artifact
and the runtime fails there rather than mid-stream.
