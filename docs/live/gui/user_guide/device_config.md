# Device and precision

The inferencer picks a device from the runtime it finds, and a precision from the artifact. Both
can be overridden, and knowing which is which avoids the common "it runs but slowly" case.

## Device strings

| String | Means | Available when |
|---|---|---|
| `cpu` | CPU execution | always |
| `0`, `1`, … | GPU index | the runtime sees a CUDA (or Intel, or Ascend) device |
| `npu` | the NPU exposed by the runtime | OpenVINO on Intel, CANN on Ascend |
| `auto` | the fastest device the runtime reports | always |

The GUI lists only the devices the installed runtime can reach — an ONNX CPU build shows one
row, an ONNX GPU build shows the GPUs.

## Precision

| Setting | Effect | Where it helps |
|---|---|---|
| `fp32` | full precision | correctness checks, CPU |
| `fp16` | half precision | GPU and NPU, usually no accuracy cost |
| `int8` | quantised | edge devices, needs a calibrated artifact |

Precision is a property of the **artifact**, not only of the runtime: an FP32 ONNX file will not
run as INT8 because `--bits 8` was passed. Export with the precision you intend to deploy.

## Fallbacks

When the requested device or precision is unavailable, the run does not silently downgrade:

- an unavailable device fails at start-up with the list of available ones;
- an artifact whose precision the runtime cannot execute fails before the first frame.

That is deliberate — a run that silently falls back to the CPU is worse than one that says it
cannot use the GPU, because the first looks like it works.

## Checking what a run uses

The inferencer prints the engine, device and precision it selected in the first lines of the run
log, and the same values are in `results/inference/records.json` under the run metadata. If they
are not what you expected, the selection — not the model — is the thing to look at.
