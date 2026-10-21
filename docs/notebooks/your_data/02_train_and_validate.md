# 2. Train and validate from Python

Drive a short training run from a notebook or a script: open the project, train, read the
records it writes, validate. The project is the one you labelled in
[the previous page](01_annotate_and_export.md).

## Opening the project

```python
from animalposetracker.project.api import AnimalPoseTrackerProject

project = AnimalPoseTrackerProject()
project.load_project_config("projects/mice-me-AnimalRTPose-N-demo/project.yaml")
```

Everything the run needs is already in the project: the keypoints and classes from the
dataset descriptor, the model family from `project.yaml`, the epochs and batch size from
`configs/runtime.yaml`.

## A short run

```python
project.train()
```

`train()` blocks while the configured training runs — the same engine as
`animalpose-cli train`, so a notebook cell and a terminal produce identical runs. For a
first pass on your own labels, set a small budget in `configs/runtime.yaml` first:

```yaml
train:
  models:
    AnimalRTPose:
      training:
        epochs: 30
        batch_size: 8
```

Early stopping can end the run before the epoch budget; that is a normal, successful exit
(exit code 0), not a crash. `early_stopping_patience: 0` disables it.

## Reading `training.jsonl`

One record per epoch lands in `results/train/training.jsonl` — losses, learning rate,
validation metrics. Reading it while the run is alive is the intended way to watch:

```python
import json

for line in open("projects/mice-me-AnimalRTPose-N-demo/results/train/training.jsonl"):
    record = json.loads(line)
    if record["event"] == "epoch":
        print(record["epoch"], record["metrics"])
```

Each line is a structured event — `event`, `epoch`, `metrics` and friends — so filter on
`event` before reading the numbers. The same metrics are plotted at the end in `results/train/results.png`. What to look for:
the box/keypoint losses falling and flattening, and the validation metric no longer
improving for `early_stopping_patience` epochs before the stop.

## Validating

```python
project.evaluate()
```

This validates `results/train/weights/best.pt` — the best checkpoint, not the last epoch —
and writes `results/val/metrics.json`: AP, AR, PCK, AUC, EPE, with the per-keypoint
breakdown. With a handful of labelled frames the absolute number means little; what matters
is whether it moves when you add labels. That loop — label more of what fails, retrain from
`best.pt`, re-validate on the same split — is the refinement cycle, and it is the whole
point of doing this from Python: the three calls above in a loop are the workflow.

```python
project.resume()  # continue from last.pt, or pass a checkpoint path
```

## Where to go next

- [3. Predict and export](03_predict_and_export.md) — the trained model on new material.
- [Training with PyTorch](../../pytorch/user_guide.md) — the training configuration keys in
  detail.
