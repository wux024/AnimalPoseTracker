# Quickstart on the built-in Tri-Mouse dataset

The full 2D loop on a packaged dataset descriptor, from `create` to an exported artifact.
Everything runs from a terminal; a GPU makes it a few minutes, a CPU longer.

## What you need

- The package installed with the `training` extra ([installation](../../installation.md));
- a local copy of the Tri-Mouse data. The packaged descriptor
  (`animalposetracker/cfg/datasets/trimouse.yaml`) carries the keypoints, skeleton and class
  names but not the images — point `--dataset-root` at your copy, or place the files under
  `<project>/datasets/` after creation.

## Steps

### 1. Create the project

```bash
animalpose-cli create --dataset trimouse --dataset-root /path/to/trimouse \
    --workspace projects --name demo --worker me \
    --model AnimalRTPose --scale N --date demo
cd projects/demo-me-AnimalRTPose-N-demo
```

The descriptor sets 12 keypoints and three classes (`mus1`, `mus2`, `mus3`). The project
directory holds `project.yaml` plus `configs/{dataset,model,runtime}.yaml`; the annotation
format (COCO JSON or YOLO TXT) is detected from the dataset root.

### 2. Check that the project sees the data

```bash
animalpose-cli load .
```

`load` prints the keypoint count, the class list, the instances per split and the detected
annotation format. A mismatch fails here, not during training.

### 3. Train

```bash
animalpose-cli train --config configs/runtime.yaml --epochs 50 --device 0
```

Watch `results/train/training.jsonl` while it runs. Early stopping may end the run before
`--epochs`; that is a normal completion. The best checkpoint lands in
`results/train/weights/best.pt`.

### 4. Validate

```bash
animalpose-cli val --config configs/runtime.yaml --weights results/train/weights/best.pt
```

`results/val/metrics.json` holds AP, AR, PCK, AUC and EPE. The OKS sigmas come from the
dataset descriptor, so numbers are comparable across runs on the same data.

### 5. Predict on new material

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/ --conf 0.25
```

`results/predict/predictions.json` records the keypoints per frame; the rendered video is
written beside it. Without `--source`, the worker predicts the configured `test` split.

### 6. Export

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx
```

The artifact and its sidecar (preprocessing metadata) land under `results/export/`. The
[export matrix walkthrough](../extra/01_export_matrix.md) goes through the other formats and
what each one needs.

## Where to go next

- [User guide](../../main-workflows/user-guide.md) — every step above with the reasoning
  spelled out.
- [Tracking and filtering](04_tracking_and_filtering.md) — identities and smoothing on the
  videos you just predicted.
