# Training with PyTorch

The trainer is the same for both families: one loop, one checkpoint format, one set of
records. This page covers what it does and which parts you can change without touching it.

## Using the trainer

Through the CLI:

```bash
animalpose-cli train --config configs/runtime.yaml \
    --epochs 200 --batch-size 8 --image-size 640 --device 0 --num-workers 8
```

Through Python:

```python
from animalposetracker.project.api import AnimalPoseTrackerProject

project = AnimalPoseTrackerProject()
project.load_project_config("projects/demo-me-AnimalRTPose-N-demo/project.yaml")
project.train()          # reads configs/runtime.yaml; no per-call arguments
```

Both paths build the same `TrainingConfig`, call the same engine, and write the same files.
The Python call takes no overrides — edit `configs/runtime.yaml` (or use the CLI) for a
one-off value. A run started one way can be resumed the other.

## Major concepts

### Profiles

`animalposetracker/cfg/training.yaml` holds two layers: `training.shared` with the values both
families use, and `training.models.<Family>` with only what differs. A project's
`configs/runtime.yaml` starts as a snapshot of the selected profile, so every key is visible
and editable.

### Optimizer

`--optimizer auto` picks per family: MuSGD for the one-stage models, AdamW for the top-down
ones. Naming an optimizer explicitly overrides the choice; the learning-rate schedule is
derived from `--learning-rate` unless a key in `runtime.yaml` says otherwise.

### EMA and AMP

An exponential moving average of the weights is kept alongside the raw model and evaluated at
each validation; `best.pt` is whichever of the two scored better. Mixed precision is a training
setting — on a device without tensor cores it degrades to full precision rather than failing.

### Freezing and resuming

- `--freeze N` freezes the backbone for the first N epochs, then unfreezes. Useful when the new
  data is small and the checkpoint already knows the domain.
- `--pretrained <path>` starts a new run from a checkpoint; bare `--pretrained` takes the
  profile's automatic one, `--no-pretrained` starts from scratch.
- `--resume path` continues an interrupted run: optimizer state, epoch counter and EMA are
  restored, so the schedule continues rather than restarting.

`--pretrained` and `--resume` are not the same thing: the first starts a new run from a model,
the second continues an interrupted one. (`train --weights` exists too, but only for
`--validate-only` — it picks the checkpoint to score, not the one to learn from.)

### Early stopping

`early_stopping_patience` counts epochs since the best validation (`epoch - best_epoch`), so it
means the same thing at any `validation_interval`. `0` disables it.

An exhausted patience is a normal completion: exit code 0, `best.pt` final. That is what makes
`train && val && export` safe in a script.

### Distributed runs

`--device 0,1` trains on two GPUs with DistributedDataParallel. Nothing else changes: the batch
size stays per device, the learning rate is scaled by the nominal batch, and the checkpoint is
written once, by rank 0.

## Where outputs go

| Path | Contents |
|---|---|
| `results/train/weights/best.pt` | best validation checkpoint |
| `results/train/weights/last.pt` | final epoch, for `--resume` |
| `results/train/training.jsonl` | one record per epoch: losses, learning rate, metrics |
| `results/train/results.png` | the curves, re-written at the end |
| `results/val/metrics.json` | AP, AR, PCK, AUC, EPE and the per-keypoint breakdown |

`training.jsonl` is append-only within a run, so an interrupted run leaves the epochs it
finished; `--resume` appends to the same file.

## API

| Object | Purpose |
|---|---|
| `animalposetracker.training.cli` | the `train` command and its flags |
| `animalposetracker.training.config.TrainingConfig` | the resolved settings for one run |
| `animalposetracker.training.engine` | the epoch loop, EMA, AMP, checkpointing |
| `animalposetracker.training.profiles` | per-family defaults and scale mapping |
| `animalposetracker.training.checkpoint` | save, load, resume |

## Changing the training recipe

Three levels, in increasing order of commitment:

1. **A flag** — `--epochs`, `--batch-size`, `--learning-rate`: one run, nothing written.
2. **`configs/runtime.yaml`** — durable for the project, visible in the file.
3. **`cfg/training.yaml`** — changes the default for every new project; a package edit, and it
   needs the tests to stay green.

Only the first two are expected in normal use. The third is for developing the toolbox itself,
and `tests/test_config_schema.py` checks that the schema and the defaults stay aligned.
