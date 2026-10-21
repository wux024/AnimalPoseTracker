# AnimalPoseTracker User Guide

This page is the whole workflow, phase by phase. It assumes the install is done
([Installing AnimalPoseTracker](../installation.md)); everything here runs from a project
directory.

## Getting started

### Command line (the route this page follows)

`animalpose-cli` exposes every stage as a subcommand — `create`, `load`, `train`, `val`,
`predict`, `infer`, `export`, `calibrate`, `pose3d`, `pose3d-render`. Each takes `--help`,
and each reads the project's `configs/runtime.yaml` unless a flag overrides it for that run.

```bash
animalpose-cli --help
animalpose-cli train --help
```

### Desktop applications

`animalposetracker`, `animalpose-annotator` and `animalpose-inferencer` drive the same
stages through windows and write the same files. A run started in the GUI can be continued
from the CLI and vice versa. See [Desktop applications](../gui/PROJECT_GUI.md).

## Workflow

### Phase 1 — Project setup

#### (A) Create a New Project

##### Overview

`create` writes a project directory and the four configuration files that describe it. The
directory name is built from the arguments — `name-worker-model-scale-date` — so the model
and the run date are visible without opening anything.

The keypoint and class layout comes from one of three places:

- `--dataset <name>` — a **packaged descriptor**, resolved from `animalposetracker/cfg/datasets/`
  (27 YAML files ship, each one registered by name and headed with its source and download link);
- `--dataset <path>` — **your own YAML**, carrying the same keys;
- `--keypoints 12 --classes 2 --class-names pup,mus` — the layout **on the command line**, with no
  file at all. `--keypoint-names` and `--skeleton 0-1,1-2` are optional, and `--no-visible` makes
  the keypoints x, y only.

Whichever you pick, the descriptor supplies what describes the data rather than the model: keypoint
names and count, class names, the skeleton, the OKS sigmas and the split folders.

`--name` is the only flag you have to give. `--workspace` defaults to the current directory,
`--worker` to `user`, the model to `AnimalRTPose` at scale `N`, and the date to now.

Neither the flags nor the form can say which keypoint mirrors onto which, so a project created
without a descriptor carries no `flip_idx` and training leaves horizontal flips off. Put the
table in a dataset YAML when the keypoints have left/right pairs.

##### Code example

```bash
animalpose-cli create --dataset trimouse --workspace projects --name demo --worker me \
    --model AnimalRTPose --scale N --date demo
```

The same call for the top-down family:

```bash
animalpose-cli create --dataset trimouse --workspace projects --name demo --worker me \
    --model AnimalViTPose --scale B --date demo
```

On your own images, with no descriptor file at all:

```bash
animalpose-cli create --keypoints 12 --classes 3 --class-names mus1,mus2,mus3 \
    --skeleton 0-1,0-2,1-3 --workspace projects --name demo --worker me --date demo
```

`create` stops at the project: identity, `configs/`, and the dataset layout. Training values
are `train`'s business, or an edit to `configs/runtime.yaml` — which `create` fills with the
model profile's defaults, so a plain `animalpose-cli train --config ...` already works.

##### Output & directory structure

```text
projects/demo-me-AnimalRTPose-N-demo/
├── project.yaml                  identity: name, worker, model, scale, date, flags
├── configs/
│   ├── dataset.yaml              keypoints, classes, skeleton, OKS sigmas, split folders
│   ├── model.yaml                backbone, head, scale, preprocessing
│   ├── runtime.yaml              every runtime parameter, sectioned by flow
│   └── cameras.yaml              only in projects created with --pose3d
├── datasets/                     links or copies of the descriptor's split folders
├── sources/                      raw images, videos, extracted frames
├── pretrained/                   cached pretrained checkpoints
└── results/                      created by the stages as they run
```

##### API Docs

- `animalposetracker.project.cli` — the `create` and `load` commands.
- `animalposetracker.project.api.AnimalPoseTrackerProject` — `create_new_project` and the
  project methods behind every stage.
- `animalposetracker.cfg` — descriptor resolution and the model registry.

#### (B) Configure the Project

##### Overview

Settings live in four files, split by what they describe, so a change lands in the file that
owns it rather than in one growing blob:

| File | Holds |
|---|---|
| `project.yaml` | Identity only: project name, worker, model type/scale, date, sources, `pose3d` / `public_dataset` flags |
| `configs/dataset.yaml` | Dataset paths and the dataset's shape — class names, keypoint names, counts, skeleton, flip table, OKS sigmas |
| `configs/model.yaml` | The network graph: scale variants, SimCC head geometry, input normalization |
| `configs/runtime.yaml` | Every runtime parameter, sectioned by flow (`common` / `train` / `predict` / `export`) |

##### Code example

Edit `configs/runtime.yaml` for a durable change:

```yaml
train:
  shared:
    training:
      early_stopping_patience: 30
  models:
    AnimalRTPose:
      training:
        epochs: 200
        batch_size: 8
        image_size: 640
```

The nesting is the file's own shape: values shared by both families sit under `train.shared`,
and per-family values under `train.models.<Family>`. A flat `train: epochs: 200` is not a key the
resolver accepts — [the configuration reference](../pytorch/pytorch_config.md) lists every path.

Override it for one run on the command line:

```bash
animalpose-cli train --config configs/runtime.yaml --epochs 50 --batch-size 4
```

The flag wins for that run and is never written back, so a one-off experiment leaves the
project untouched.

##### Precedence

1. a command-line flag,
2. the project's `configs/runtime.yaml`,
3. the model profile's defaults in `animalposetracker/cfg/training.yaml`.

`training.shared` holds the values both model families share; each profile under
`training.models` carries only what differs. The full key reference is generated from the
schema: [Configuration reference](../pytorch/pytorch_config.md).

##### API Docs

- `animalposetracker.config.resolver` — precedence and validation.
- `animalposetracker.config.schema` — the registered keys and their types.

### Phase 2 — Data preparation

#### (C) Prepare Frames and Splits

##### Overview

Training reads frames from the split folders declared in `configs/dataset.yaml`:

```text
datasets/
├── images/{train,val,test}/
├── labels/{train,val,test}/      YOLO-Pose txt
└── annotations/                  COCO json, when that format is used
```

Frame extraction is a desktop step: the workspace's **Manage** tab samples a video
automatically — `kmeans` clustering by default, or `uniform` spacing — and writes frames into
`sources/extracted/`. Manual capture lives in the annotator (`Start Label Frames`). The CLI has
no `extract` subcommand — move the frames into the split folders yourself, or point
`dataset.yaml` at wherever they already are.

##### Code example

A descriptor that points at an existing layout:

```yaml
path: /data/mouse
train: images/train
val: images/val
test: images/test
```

If the frames live outside the project, keep `path` absolute; nothing is copied.

##### API Docs

- `animalposetracker.utils.extract_frame` — the extraction helper the GUI calls.
- `animalposetracker.project.files` — split directory resolution.

#### (D) Label Frames

##### Overview

`animalpose-annotator` draws boxes, assigns classes, places keypoints and records the
skeleton. It writes either format:

- **YOLO-Pose TXT** — one `.txt` per image, beside the image in `labels/<split>/`;
- **COCO Keypoints JSON** — one JSON per split, under `annotations/`.

Both are read directly at training time; nothing converts between them.

##### Code example

```bash
animalpose-annotator
```

Then open the project directory. The annotator lists the frames it finds under `sources/`
and the split folders, and saves labels where the project expects them.

##### API Docs

- `animalposetracker.gui.annotatorpage` — the annotation window.
- `animalposetracker.data.coco` — COCO read/write.
- `animalposetracker.data.pose` — YOLO-Pose read/write.

#### (E) Check Annotated Frames

##### Overview

Before training, `load` validates the project and prints what the trainer will see:

- the keypoint count (`kpt_shape`) against the labels;
- the class list and the number of instances per split;
- the annotation format, detected from the file layout unless `annotation_format` pins it;
- the OKS sigmas and the flip table, when present.

A mismatch is reported by name — a label file with 12 keypoints against a `kpt_shape: [17, 3]`
descriptor fails here, not an hour into training.

##### Code example

```bash
animalpose-cli load projects/demo-me-AnimalRTPose-N-demo
```

##### API Docs

- `animalposetracker.project.cli.load` — project validation and summary.
- `animalposetracker.data.pose.validate` — the per-split checks.

### Phase 3 — Training & evaluation

#### (F) Build the Training Inputs

##### Overview

There is no separate "create dataset" step. The dataloader reads the split folders on every
run, applies the model's preprocessing and augmentation, and caches decoded images.

What the pipeline does, and where it is configured:

| Step | Behaviour | Key |
|---|---|---|
| Letterbox | keeps the aspect ratio, pads to the square input | `image_size` |
| Augmentation | mosaic, mixup, flips, colour jitter; the flip table keeps left/right keypoints consistent | `train.augmentation` |
| Top-down crop | only for AnimalViTPose: each instance is cropped from its box with padding, then resized | `model.crop_padding` |
| Caching | decoded images are cached when the dataset fits in memory | `train.cache` |

##### Code example

```yaml
# configs/dataset.yaml
kpt_shape: [12, 3]
flip_idx: [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11]
kpt_oks_sigmas: [0.026, 0.025, 0.025, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035]
```

##### API Docs

- `animalposetracker.data.pose` — the dataset and dataloader.
- `animalposetracker.data.augment` — augmentation and the flip table.
- `animalposetracker.data.topdown` — the crop-and-resize path for AnimalViTPose.

#### (G) Train the Network

##### Overview

`train` builds the model from `configs/model.yaml`, loads the splits from
`configs/dataset.yaml`, and writes everything under `results/train/`.

The knobs that matter first:

| Flag | Meaning |
|---|---|
| `--epochs` | upper bound; early stopping can end the run sooner |
| `--batch-size` | per-device batch |
| `--image-size` | square input; both families are trained at one size |
| `--device` | `0`, `0,1` for multi-GPU, or `cpu` |
| `--num-workers` | dataloader processes |
| `--freeze` | freeze the backbone for the first N epochs |
| `--optimizer` | `auto` picks MuSGD for the one-stage family, AdamW for the top-down |
| `--pretrained` | bare for the profile's automatic checkpoint, a path to start from a local one, `--no-pretrained` for from scratch |
| `--resume` | continue an interrupted run from its `last.pt` |

Early stopping counts **epochs since the best validation** (`epoch - best_epoch`), so it
means the same thing at any validation interval. `early_stopping_patience: 0` disables it.

##### Code example

```bash
animalpose-cli train --config configs/runtime.yaml \
    --epochs 200 --batch-size 8 --image-size 640 --device 0 --num-workers 8
```

Continue a run that was interrupted:

```bash
animalpose-cli train --config configs/runtime.yaml --resume results/train/weights/last.pt
```

Start from a checkpoint with a frozen backbone:

```bash
animalpose-cli train --config configs/runtime.yaml \
    --pretrained projects/old/results/train/weights/best.pt --freeze 10
```

##### What it writes

- `results/train/weights/best.pt` — the best validation checkpoint;
- `results/train/weights/last.pt` — the final epoch, for `--resume`;
- `results/train/training.jsonl` — one record per epoch: losses, learning rate, metrics;
- `results/train/results.png` — the same numbers plotted.

##### Exit codes

`train` exits `0` both when it reaches `epochs` and when early stopping ends it. An external
interrupt (`SIGINT`/`SIGTERM`, or the `.animalposetracker_training_stop` marker) exits `130`;
an error exits `1`. That makes `val` and `export` safe to chain in a `set -e` script.

##### API Docs

- `animalposetracker.training.cli` — the command and its flags.
- `animalposetracker.training.engine` — the epoch loop, EMA and AMP.
- `animalposetracker.training.profiles` — the per-model defaults.

#### (H) Evaluate the Trained Network

##### Overview

`val` runs the native validator on the configured validation split. It reports:

| Metric | What it measures |
|---|---|
| `coco/AP` | keypoint average precision over OKS thresholds |
| `coco/AR` | average recall at the same thresholds |
| `PCK` | percentage of correct keypoints within a fraction of the instance scale |
| `AUC` | area under the PCK curve |
| `EPE` | end-point error, in pixels |

AnimalViTPose validation does **not** use flip test-time augmentation by default, so the
numbers match what the shipped decoder produces. Set `validation.flip_tta: true` to re-enable
it — expect higher numbers than the deployment path.

##### Code example

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt
```

##### What it writes

`results/val/metrics.json` — the aggregate metrics plus the per-keypoint breakdown where it
applies, and the same decoder settings the prediction path uses.

##### API Docs

- `animalposetracker.evaluation.metrics` — OKS, PCK, AUC and EPE.
- `animalposetracker.evaluation.animalrtpose` — the one-stage adapter.
- `animalposetracker.evaluation.topdown` — the top-down adapter.

### Phase 4 — Analysis

#### (I) Analyze New Videos

##### Overview

`predict` runs the trained model on new material. Three input modes:

- `--split test` (or `val`) — the configured split, boxes read from the annotations;
- `--source <path>` — an image, a folder, a video file or a glob;
- neither — the configured `test` split, then `val`.

AnimalRTPose predicts on the full image. AnimalViTPose crops each instance from its box, runs
the ViT on the crop, and maps the SimCC points back to the source frame.

##### Code example

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --split test \
    --conf 0.25 --iou 0.7 --max-detections 300
```

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/ \
    --conf 0.4 --no-show-boxes
```

##### What it writes

`results/predict/`:

- `predictions.json` — per-frame keypoints with scores;
- `prediction_metadata.json` — the artifact, format, split and tracker the run used;
- rendered images or videos, when the run draws them.

##### API Docs

- `animalposetracker.prediction.cli` — the command and its flags.
- `animalposetracker.prediction.backends` — per-model inference and rendering.

#### (J) Filter Pose Data

##### Overview

Two post-processing stages run on the pose records, so a sequence can be re-processed
without touching the network:

- **tracking** assigns identities across frames: `bytetrack`, `botsort`, `ocsort`,
  `deepocsort`, `fasttrack`, `tracktrack`;
- **filtering** smooths each track: `median` for spikes, `savgol` for smooth drift.

Both read `predictions.json` and write a sibling record file, leaving the original intact.

##### Code example

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/ \
    --tracker bytetrack --track-track-buffer 45 \
    --pose-filter median --filter-window 7 --filter-confidence 0.3 --filter-max-gap 5
```

##### What it writes

- `tracked_predictions.json` — identities added;
- `predictions-filter.json` — smoothed keypoints, with the parameters that produced them.

##### API Docs

- `animalposetracker.tracking` — the six trackers and their flags.
- `animalposetracker.postprocessing.filters` — median and Savitzky–Golay.

#### (K) Plot and Summarize Results

##### Overview

There is no separate 2D trajectory plotting tool. The summaries that ship are:

- training curves — `results/train/results.png`, written by `train`;
- validation metrics — `results/val/metrics.json`, including the per-keypoint breakdown;
- 3D trajectories — `animalpose-cli pose3d-render --plot`, for projects with a camera rig.

For anything else, the records are plain JSON; read them with the API and plot with whatever
you already use.

##### Code example

```bash
animalpose-cli pose3d-render --config configs/runtime.yaml --plot
```

##### API Docs

- `animalposetracker.evaluation.metrics` — the numbers behind the curves.
- `animalposetracker.pose3d.export` — CSV, summary and trajectory plot writers.

#### (L) Create Labeled Videos

##### Overview

When `predict` runs on a video, it renders the skeleton back onto the frames. The drawing is
configurable, and the toggles are the same ones the inferencer uses.

| Flag | Effect |
|---|---|
| `--show-boxes` / `--no-show-boxes` | draw instance boxes |
| `--show-skeletons` / `--no-show-skeletons` | draw the skeleton |
| `--show-keypoints` / `--no-show-keypoints` | draw individual points |
| `--show-conf` / `--no-show-conf` | print the score beside each instance |
| `--line-width`, `--point-radius` | stroke and marker size |

##### Code example

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/clip.mp4 \
    --show-skeletons --show-keypoints --line-width 3 --point-radius 4
```

##### What it writes

`results/predict/videos/` — the source video re-encoded with the overlay drawn.

##### API Docs

- `animalposetracker.rendering` — the drawing primitives.

### Phase 5 — Refinement (optional)

#### (M) Find and Review Failure Cases

##### Overview

There is no automatic outlier mining. The signal comes from three places:

1. **per-keypoint metrics** in `metrics.json` — a landmark with a much lower AP or a much
   higher EPE is where the model is weak;
2. **the rendered videos** — a skeleton that slides off the animal marks frames worth
   relabelling;
3. **the PR curve and confusion matrix** — written with the validation run, they show whether
   the error is localization or classification.

Take the frames those point at, and relabel them with `animalpose-annotator`.

##### Code example

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --output-dir results/val-check
```

Then open `results/val-check/metrics.json` and sort the per-keypoint entries.

##### API Docs

- `animalposetracker.evaluation.metrics` — per-keypoint AP and EPE.
- `animalposetracker.gui.annotatorpage` — relabelling.

#### (N) Refine and Retrain

##### Overview

The refinement loop is: add the frames, retrain from the current weights, compare on the same
split.

- **Add frames** to the same split folders, or create a new split and point
  `configs/dataset.yaml` at it.
- **Retrain from `best.pt`** with `--pretrained`, so the new run starts from what already works
  rather than from the profile checkpoint.
- **Freeze the backbone** for the first epochs with `--freeze N` when the new frames are few.
- **Compare** `metrics.json` before and after — same split, same decoder, so the numbers are
  directly comparable.

##### Code example

```bash
animalpose-cli train --config configs/runtime.yaml \
    --pretrained results/train/weights/best.pt --freeze 10 --epochs 60
animalpose-cli val --config configs/runtime.yaml --weights results/train/weights/best.pt
```

##### API Docs

- `animalposetracker.training.cli` — `--pretrained`, `--freeze`, `--epochs`.
- `animalposetracker.training.checkpoint` — how a checkpoint is resumed.

### Phase 6 — Deployment

#### (O) Export the Trained Network

##### Overview

`export` turns a checkpoint into a deployment artifact and a sidecar JSON. One flag decides the
format; [Deployment formats](../recipes/OpenVINO.md) lists what each one needs.

##### Code example

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx --dynamic --opset 17
```

`results/export/` holds the artifact and `<artifact>.json`. The sidecar carries what the graph
cannot: the head, the input size, the colour order, the keypoint shape and — for SimCC heads —
the normalisation and crop padding.

##### API Docs

- `animalposetracker.export` — the format matrix and the per-backend exporters.

#### (P) Run the Artifact Live

##### Overview

`infer` reads no project configuration: the artifact, its sidecar, and an optional YAML. It
takes a camera index, a video file or a stream URL, and `--records` writes the same record
schema `predict` writes.

```bash
animalpose-cli infer --poser-weights results/export/best.onnx --source 0 --records live.json
```

A SimCC artifact also needs `--detector-weights`; without it the inferencer refuses the model.

##### API Docs

- `animalposetracker.inference` — the engine and the stream pipeline.

## Resources and further reading

### The usage demo

[the single-animal quick guide](../quick-start/single_animal_quick_guide.md) runs the whole 2D workflow on the Tri-Mouse descriptor and lists every
file each step leaves behind.

### 3D tracking

[Multi-camera 3D](../Overviewof3D.md) covers calibration, association, triangulation and
rendering; the Python API is in [Pose analysis workflows](../recipes/post.md).

### Additional helper functions

- [Configuration reference](../pytorch/pytorch_config.md) — every registered key.
- [Deployment formats](../recipes/OpenVINO.md) — the export matrix and per-format extras.
- [Desktop applications](../gui/PROJECT_GUI.md) — the same stages through windows.
