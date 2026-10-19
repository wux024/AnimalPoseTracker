# Pretrained weights and public benchmark datasets

This page is about what ships as weights and what ships as data descriptors. It is **not** a
model zoo: there is no hosted collection of task-specific models, and no SuperAnimal-style
generalist checkpoint.

## What this page is

Two things are packaged:

1. **Pretrained checkpoints** for the model profiles, downloaded on first use into the project's
   `pretrained/` directory;
2. **Dataset descriptors** for public benchmarks, which supply keypoints, skeleton, OKS sigmas
   and split folders — not the images.

## Pretrained checkpoints

`animalposetracker/cfg/models/animalrtpose.yaml` carries one URL per scale:

| Scale | Weights |
|---|---|
| N / S / M / L / X | full pose-model checkpoints, one per scale |

`animalposetracker/cfg/models/animalvitpose.yaml` carries the MAE backbone initialization for
each variant (Small / Base / Large / Huge), which is pretraining for the backbone, not a pose
model — the head is trained on your data.

### Download timing

Nothing downloads at install time. A checkpoint is fetched the first time a run needs it, into
`<project>/pretrained/`, and reused after that. An offline machine can pre-place the files there
with the expected names, or start from `--weights` instead.

## Public benchmark descriptors

`animalposetracker/cfg/datasets/` holds 29 YAML files. 27 are registered by name and can be
passed to `create --dataset <name>`; `realworldperson.yaml` and `twoperson.yaml` are
file-only, so pass their path instead.

| Kind | Examples |
|---|---|
| Animal pose benchmarks | `animalpose`, `ap10k`, `apt36k`, `atrw`, `horse10` |
| Single-species sets | `trimouse`, `topviewmouse`, `fish`, `fly`, `locust` |
| Human pose | `coco-pose`, `lsp`, `mhp`-style sets where present |

Each descriptor carries the data's shape, not the data: the split folders point at wherever the
images live, and the download (when the descriptor has one) is a one-time step.

### Training on a benchmark

```bash
animalpose-cli create --dataset ap10k --workspace projects --name ap10k --worker me \
    --model AnimalRTPose --scale S --date run1
animalpose-cli train --config projects/ap10k-me-AnimalRTPose-S-run1/configs/runtime.yaml
animalpose-cli val --config projects/ap10k-me-AnimalRTPose-S-run1/configs/runtime.yaml \
    --weights projects/ap10k-me-AnimalRTPose-S-run1/results/train/weights/best.pt
```

The metric is comparable with the README's table only if the split and the OKS sigmas match;
both come from the descriptor, so use the same one.

## Fine-tuning

Starting from a checkpoint rather than from scratch is one flag:

```bash
animalpose-cli train --config configs/runtime.yaml \
    --weights pretrained/best.pt --freeze 10 --epochs 60
```

`--freeze 10` keeps the backbone fixed for ten epochs, which is usually enough to adapt the head
to a new keypoint layout. The full recipe is in [Fine-tuning](recipes/fine_tuning.md).

## What is not here

- **No task-specific model collection.** The checkpoints are the published model families; a
  model trained on your species is yours to keep and share.
- **No generalist multi-species checkpoint.** The descriptors let you train on a benchmark, but
  each project is one dataset with one keypoint set.
- **No dataset hosting.** Descriptors carry URLs where the licence allows; the files stay with
  their original hosts.

## Failure modes

| Symptom | Cause |
|---|---|
| `create` cannot resolve a dataset name | the YAML is file-only (`realworldperson`, `twoperson`) — pass a path |
| download stalls on the first run | the descriptor's host is unreachable; pre-place the file in `pretrained/` |
| AP several points below the table | different split, or missing `kpt_oks_sigmas` |
