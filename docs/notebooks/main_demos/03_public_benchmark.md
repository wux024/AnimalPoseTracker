# Training on a public benchmark

Take a public-benchmark descriptor (AP-10K here), train a scale, and produce a number you
can line up with the published table. The point of this walkthrough is not the training
loop — it is knowing exactly which choices decide whether your number is comparable.

## What "comparable" means

A validation number is only comparable when three things match the published row:

1. **the split** — `val` always scores the configured validation split;
2. **the OKS sigmas** — one tolerance per keypoint, from the dataset descriptor;
3. **the decoder** — the shipped decoder, not a separate evaluation harness.

Change any of the three and the gap you measure is the harness, not the model.

## Steps

### 1. Create from the descriptor

```bash
animalpose-cli create --dataset ap10k --dataset-root /path/to/ap10k \
    --workspace projects --name bench --worker me \
    --model AnimalRTPose --scale N --date bench
cd projects/bench-me-AnimalRTPose-N-bench
```

`animalposetracker/cfg/datasets/` carries 27 descriptors — AP-10K, APT-36K, Animal Kingdom,
Horse-10, StanfordExtra and others. Each one opens with its source and download link, and
fixes the keypoint list, the skeleton, the class names and the split layout. The images
themselves are yours to download from the benchmark's authors; `create` never fetches data.

### 2. Check the OKS sigmas

```bash
grep -A20 kpt_oks_sigmas configs/dataset.yaml
```

AP-10K ships per-keypoint sigmas, and the descriptor carries them into the project. If a
descriptor omits `kpt_oks_sigmas`, the trainer warns and falls back to a uniform `1 / K` —
AP stays computable but is no longer comparable with published numbers.

### 3. Train at the benchmark's input size

```bash
animalpose-cli train --config configs/runtime.yaml \
    --epochs 300 --batch-size 96 --image-size 640 --device 0
```

Match the input size of the row you want to compare with — the published AnimalRTPose
accuracy table is 640×640, the AnimalViTPose table is 256×256. Watch
`results/train/training.jsonl`; `results/train/results.png` is written at the end.

### 4. Validate

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --output-dir results/val-bench
```

`metrics.json` carries `coco/AP`, `coco/AR`, PCK, AUC and EPE with the per-keypoint
breakdown. AP is the headline number to line up with the table.

### 5. Read it against the published row

The README's model tables were measured once, on fixed hardware, with this toolbox's own
decoder. If your split, sigmas, input size and decoder match, your AP should land near the
published row; a large gap means one of the four drifted. Check in that order: split →
sigmas → image size → decoder.

Numbers from other frameworks (mmpose, ultralytics) are scored by *their* harnesses on
*their* splits — they are context, not a regression target. The only fair comparison inside
this toolbox is two `val` runs on the same project.

## Where to go next

- [Benchmarking and evaluation](../../pytorch/benchmarking.md) — metric definitions and the
  latency side.
- [Pretrained weights and datasets](../../ModelZoo.md) — scales, variants and what ships.
