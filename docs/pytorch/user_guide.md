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
The Python call takes no overrides — edit `configs/runtime.yaml` or use the CLI, which records
each flag it was given. A run started one way can be resumed the other.

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
each validation; `best.pt` is whichever of the two scored better.

`amp` takes `false`, `true` or a precision name (`fp16`, `bf16`, `fp32`). With `true`, startup
runs one forward on synthetic input at fp16 → bf16 → fp32 and keeps the first precision whose
output stays finite: fp16 saturates at 65504, bf16 needs Ampere or newer. When the probe falls
back to fp32, the batch is halved and the gradient accumulation doubled, so the nominal batch
and the optimizer step stay unchanged. Validation forward runs under autocast at the resolved
precision while the weights themselves stay fp32 — casting EMA weights to half in place would
round them at every validation.

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

### Non-finite recovery

A loss or metric that turns non-finite rolls the run back to `last.pt` — weights, optimizer
and schedule — and replays the epoch, up to three retries. A first epoch that has no
checkpoint yet still raises: there is nothing to roll back to.

### Batch size auto

`--batch-size auto` probes real forward/backward steps at powers of two (512 down to 8) and
keeps the largest whose measured peak memory stays under 60% of the GPU; the candidate is
never larger than the number of training images.

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

1. **A flag** — `--epochs`, `--batch-size`, `--learning-rate`: wins the run, then is recorded
   into `configs/runtime.yaml` so the next run starts from it. Run-level keys (`--project`,
   `--name`, `--exist-ok`, `--device`) describe the run and are never written back.
2. **`configs/runtime.yaml`** — durable for the project: the profile snapshot plus every
   override a run has recorded.
3. **`cfg/training.yaml`** — changes the default for every new project; a package edit, and it
   needs the tests to stay green.

Only the first two are expected in normal use. The third is for developing the toolbox itself,
and `tests/test_config_schema.py` checks that the schema and the defaults stay aligned.
