# Performance benchmarks

Latency and accuracy are separate measurements. This page is about the first: what was
measured, on which hardware, and how to reproduce a row.

## What can be measured

| Measurement | Tool | What it tells you |
|---|---|---|
| inference latency | `infer` on a recorded clip | how fast one forward pass is on your device |
| training throughput | `training.jsonl` (`epoch` duration) | how long a run will take |
| accuracy | `val` | whether the model is good enough |
| end-to-end throughput | `infer` with a camera | whether the pipeline keeps up with the frame rate |

Only latency is listed below. Accuracy is in the README's model tables and in
[Benchmarking and evaluation](pytorch/benchmarking.md).

## Published latency

AnimalRTPose on NVIDIA GPUs, milliseconds:

| Model | RTX 2080 Ti | RTX 3090 | A800 |
|---|---|---|---|
| N | 2.1 | 1.3 | 0.9 |
| S | 3.6 | 2.5 | 1.3 |
| M | 7.5 | 4.9 | 2.4 |
| L | 11.3 | 7.5 | 3.7 |
| X | 17.1 | 11.9 | 5.6 |

On edge devices:

| Model | Jetson AGX Orin | Raspberry Pi 5 + AI HAT+ | Atlas 200I |
|---|---|---|---|
| N | 5.1 | 13.0 | 15.6 |
| S | 5.0 | 22.0 | 26.5 |
| M | 6.3 | 33.2 | 53.2 |
| L | 7.5 | 49.3 | 105.7 |
| X | 9.1 | 74.1 | 162.6 |

On the workstations the toolbox was developed on (same video, same box):

| Machine | Compute unit | Inference |
|---|---|---|
| Workstation 1 | RTX 4090 | 6.0 ms |
| Workstation 1 | Intel integrated GPU | 10.0 ms |
| Workstation 1 | Intel NPU | 25.5 ms |
| Workstation 1 | CPU only | 18.2 ms |
| Workstation 2 | RTX 4080 | 2.6 ms |
| Workstation 2 | CPU only | 14.2 ms |

The full machine list, including the laptops and the Mac Mini, is in
[Hardware considerations](installation.md) (end of the page).

## Reproducing a row

Latency depends on the artifact, the engine, the input size and the precision. To reproduce a
row:

1. export the same scale at the same input size and precision;
2. run `infer` on a recorded clip long enough to average out start-up;
3. read the per-frame time from the run's output.

Expect differences from the table in the tens of percent: driver versions, clock behaviour and
thermal state all move the number. The table is an order-of-magnitude guide, not a contract.

## What the numbers do not cover

- **Pre/post-processing.** The tables measure the forward pass; decoding, drawing and writing
  add time, and on a CPU device they can dominate.
- **Multi-camera throughput.** Each camera is an independent stream; the per-frame number does
  not divide by the camera count without accounting for scheduling.
- **Training.** A training run is bounded by data loading and augmentation as much as by the
  GPU; `training.jsonl` reports the epoch duration on your machine, which is the only number
  that matters for your data.
