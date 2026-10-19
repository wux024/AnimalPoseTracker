# Inputs and outputs

## What each stage reads

- `create` — a dataset descriptor, or a registered name
- `train` — `configs/runtime.yaml`, `dataset.yaml`, `model.yaml`
- `val` — the same, plus a checkpoint
- `predict` — a checkpoint, a split or `--source`
- `infer` — an exported artifact and `infer.yaml`
- `pose3d` — one `predictions.json` per camera and `cameras.yaml`

## What each stage writes

- `results/train/` — `weights/best.pt`, `last.pt`, `training.jsonl`, `results.png`
- `results/val/` — `metrics.json`
- `results/predict/` — `predictions.json`, `prediction_metadata.json`, renders
- `results/inference/` — `records.json`, `output.avi`
- `results/pose3d/` — `poses3d.json`, `report.json`, CSV, videos
- `results/export/` — `<model>.<format>` plus the `.json` sidecar

## Directory conventions

- splits are directories under `datasets/images/`
- labels mirror the split layout, or live in `annotations/`
- nothing writes outside the project directory except an explicit `--output-dir`
