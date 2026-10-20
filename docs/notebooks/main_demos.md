# Demo walkthroughs

Four runs on packaged datasets. Each one is short enough to follow in a session and covers a
different part of the toolbox.

## What is here

| Walkthrough | Dataset | What it shows |
|---|---|---|
| [Quickstart on Tri-Mouse](main_demos/01_quickstart_cli.md) | Tri-Mouse | the full 2D loop, create to export |
| [Top-down with AnimalViTPose](main_demos/02_topdown_vitpose.md) | Tri-Mouse | boxes, crops, SimCC decoding |
| [Public benchmark](main_demos/03_public_benchmark.md) | AP-10K-style | training and reading the published metric |
| [Tracking and filtering](main_demos/04_tracking_and_filtering.md) | any sequence | identities across frames, smoothing |

## Which demo to pick

- **New to the toolbox** — start with the quickstart. It writes every file the later stages
  expect, so it doubles as a reference for what a project should look like.
- **Overlapping instances** — the top-down walkthrough, because boxes are the whole point of
  that family.
- **Comparing with a paper** — the benchmark walkthrough; it is explicit about which choices
  make a number comparable.

## Before you start

The packaged descriptors carry the keypoints, skeleton and class names — not the images.
Have a local copy of the dataset and point `--dataset-root` at it; `create` never downloads
data.
