# CoreML engine

Runs `.mlpackage` artifacts through the CoreML runtime.

## Artifact

| Property | Value |
|---|---|
| Format | `.mlpackage` |
| Install | `pip install coremltools>=8.0` |
| Devices | Apple silicon, macOS |

## When to use it

Pick this engine when the deployment machine is apple silicon and the
artifact was exported for it. The engine is implied by the artifact's extension;
`--engine` overrides the choice when a format has several runtimes.

## Limits

- export needs a Mac with coremltools
- compute units: CPU, GPU, Neural Engine, or all
- the package bundles the preprocessing it was exported with

## Checks

```bash
animalpose-cli infer --config infer.yaml
```

The first frames report the engine and device in use; a mismatch between the artifact
and the runtime fails there rather than mid-stream.
