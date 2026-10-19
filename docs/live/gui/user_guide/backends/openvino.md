# OpenVINO engine

Runs `.xml + .bin` artifacts through the OpenVINO runtime.

## Artifact

| Property | Value |
|---|---|
| Format | `.xml + .bin` |
| Install | `pip install openvino>=2024.0.0` |
| Devices | Intel CPU, GPU, NPU |

## When to use it

Pick this engine when the deployment machine is intel cpu and the
artifact was exported for it. The engine is implied by the artifact's extension;
`--engine` overrides the choice when a format has several runtimes.

## Limits

- the IR pair is written by the OpenVINO exporter
- the device string selects CPU / GPU / NPU
- FP16 is the default on GPU; INT8 needs calibration

## Checks

```bash
animalpose-cli infer --config infer.yaml
```

The first frames report the engine and device in use; a mismatch between the artifact
and the runtime fails there rather than mid-stream.
