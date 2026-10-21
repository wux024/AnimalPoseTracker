# Benchmarking and evaluation

This page covers what the numbers mean, how to produce them on your own split, and what
"reproducing a published row" actually requires.

## What is measured

Two things, separately:

- **Accuracy** — how close the predicted keypoints are, scored against the labels.
- **Latency** — how long one forward pass takes on a given device.

They are measured by different tools and are not interchangeable: a fast model that is
inaccurate is not better than a slow one that works, and `val` says nothing about speed.

## Metrics

| Metric | Definition | Sensitive to |
|---|---|---|
| `coco/AP` | mean average precision over OKS thresholds 0.50:0.95 | everything; the headline number |
| `coco/AR` | mean average recall over the same thresholds | missed instances |
| `PCK` | fraction of keypoints within a distance threshold, normalized by instance scale | coarse localization |
| `AUC` | area under the PCK curve across thresholds | the shape of the error distribution |
| `EPE` | end-point error in pixels | absolute localization, per keypoint |

AP is the one to compare across models; EPE is the one to look at when a single landmark is
off.

### OKS sigmas

AP needs one positive sigma per keypoint, set as `kpt_oks_sigmas` in the dataset descriptor.
The sigmas encode how tolerant the metric is for each landmark — a wrist and an eye do not
deserve the same tolerance.

If the key is omitted, the trainer warns and falls back to a uniform `1 / K`. That keeps AP
computable but makes it incomparable with published numbers, which use per-keypoint sigmas.

## Running a benchmark

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt \
    --output-dir results/val-benchmark
```

The run writes `metrics.json` with the aggregate numbers and the per-keypoint breakdown. What
determines whether the number is comparable:

1. **the same split** — `val` always uses the configured validation split;
2. **the same sigmas** — from the dataset descriptor;
3. **the same decoder** — the shipped decoder, not a separate evaluation harness.

Change any of the three and the number moves for reasons that have nothing to do with the
model.

## Comparing runs

Because the split, the sigmas and the decoder are fixed by the project, two `val` runs on the
same project are directly comparable — different checkpoints, different scales, different
augmentation settings.

```bash
animalpose-cli val --config configs/runtime.yaml --weights runA/best.pt --output-dir results/valA
animalpose-cli val --config configs/runtime.yaml --weights runB/best.pt --output-dir results/valB
```

To compare across model families, create both projects from the same dataset descriptor. The
top-down family will additionally need boxes at validation time; those come from the split's
annotations, so the comparison stays fair.

## Published numbers

The README carries four tables, measured once on fixed hardware:

- AnimalRTPose accuracy on AnimalPose at 640×640, per scale;
- inference latency on NVIDIA GPUs, per scale;
- latency on edge devices, per scale;
- AnimalViTPose accuracy on AP10K at 256×256, per variant.

They are measurements, not a benchmark framework: there is no script that regenerates the
tables, and the hardware is named because latency without a device is meaningless.

## Reproducing a row

To reproduce an accuracy row:

1. `create` a project on the same dataset descriptor (`animalpose` or `ap10k`);
2. train with the published scale and input size;
3. run `val` and read `coco/AP`.

Expect small differences: the split files, the augmentation schedule and the random seed all
move the last decimal. A number within a point of the table means you reproduced it; a number
several points off usually means a different split or a missing `kpt_oks_sigmas`.

To reproduce a latency row, export the artifact for that device and time a forward pass with
the same input size and precision. `val` does not measure latency.
