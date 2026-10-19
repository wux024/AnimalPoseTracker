# 🥳 Get started with AnimalPoseTracker: our key recommendations

Below we outline what you need to get started, the two model families the toolbox carries,
and the full workflow from an empty folder to a deployed model. It assumes nothing about
your species, viewpoint or camera count.

```{note}
The documentation is being expanded. The user guide and the reference pages are complete;
the demo walkthroughs are still being written and appear as short placeholders.
```

```{hint}
💡📚 If you are new to pose estimation, read *What we support* below first, then follow the
[User guide](main-workflows/user-guide.md) from Phase 1. The [single-animal quick guide](quick-start/single_animal_quick_guide.md) is the
shortest path to a trained model.
```

## Introduction

**AnimalPoseTracker** is a toolbox for markerless pose estimation of animals performing
various behaviours. It manages multiple projects; each project is identified by a name, a
worker and the creation date, and it lives in a directory that holds four configuration
files plus everything the stages write.

A project does not depend on a particular model. The same `predictions.json`, the same
tracking records and the same 3D stage serve both model families, so switching the model
changes one configuration key rather than the workflow.

## Installing AnimalPoseTracker

We do not cover installation in depth on this page. The commands, the extras and the engine
matrix are in [Installing AnimalPoseTracker](installation.md); come back here once
`animalpose-cli --version` answers.

## What we support

We are primarily a package that trains and serves pose models. There are many knobs, but the
defaults are chosen to work before they are tuned — pass only what you need to `create` and
the rest stays at the model profile's values.

### Main modes of AnimalPoseTracker

There are **two model families**, and the choice between them is the first decision a
project makes:

- **AnimalRTPose** — one stage. Boxes and keypoints come out of a single forward pass, so it
  needs no detector and no instance crop. Scales N / S / M / L / X. This is the default, and
  the right starting point for most single-animal and small-group recordings.
- **AnimalViTPose** — top-down. A box is supplied (from your labels, or from a detector in
  live inference), the instance is cropped and resized, a ViT with a SimCC head predicts the
  keypoints inside the crop, and the points are mapped back to the source image. Scales
  Small / Base / Large / Huge. This is the accuracy path when boxes are available.

To note: there is no multi-animal project mode, no bundled model zoo and no detector shipped
with the package. The one-stage family predicts instances directly; the top-down family
needs boxes, which the project reads from the split's annotations or the live inferencer
obtains from `--detector-weights`.

```{tip}
Start with AnimalRTPose on a packaged dataset, even if your goal is the top-down family. One
short run through the workflow makes the later decisions — boxes, crops, evaluation — much
easier to reason about.
```

### Additional learning resources

- [User guide](main-workflows/user-guide.md) — the workflow in five phases, one section per
  step.
- [Usage demo](quick-start/single_animal_quick_guide.md) — a full run on a built-in dataset: create, train, validate, predict,
  export.
- [Multi-camera 3D](Overviewof3D.md) — calibration, association, triangulation, rendering.
- [Pose analysis workflows](recipes/post.md) — the Python API for tracking, filtering and
  triangulation.
- [Configuration reference](pytorch/pytorch_config.md) — every registered key, generated
  from the schema.

### What you need to get started

- **A set of videos or images that span the behaviours you want to track.** Ten short clips
  across backgrounds and individuals beat one long clip of one animal: the model can only
  learn what the labels show it.
- **Ideally, a computer with a GPU.** Training runs on the CPU, but a CUDA device is what
  makes a full run practical. Inference on an exported artifact runs anywhere.
- **Labels.** Either format works, and both are read directly:
  - YOLO-Pose TXT, one file per image;
  - COCO Keypoints JSON, one file per split.

### What you DON'T need to get started

- No specific camera or video format. Colour, monochrome, phone footage and lab cameras are
  all fine — if you can see what you want to measure, the toolbox can be trained on it.
- No specific computer. The package runs on Linux, Windows and macOS; the desktop
  applications need the `gui` extra, the CLI does not.
- No detector, unless you choose the top-down family in live inference. For training and
  project prediction, boxes come from the annotations you already have.

## Workflow overview

Every project follows the same eight steps. Steps 1–7 run from the command line or the
desktop applications; step 8 needs a calibrated camera rig.

| Step | What happens | Command |
|---|---|---|
| 1 | Create the project and its four configs | `animalpose-cli create` |
| 2 | Label boxes, classes, keypoints and skeleton | `animalpose-annotator` |
| 3 | Train | `animalpose-cli train` |
| 4 | Validate on the validation split | `animalpose-cli val` |
| 5 | Predict on images, videos or the test split | `animalpose-cli predict` |
| 6 | Run an exported artifact live | `animalpose-cli infer` |
| 7 | Export to a deployment format | `animalpose-cli export` |
| 8 | Calibrate, triangulate, render 3D | `calibrate` / `pose3d` / `pose3d-render` |

You can have as many projects on one machine as you wish. Each stage resumes from the
project directory, so a run can be stopped and continued without re-deriving anything.

## Usage advice & project types

### First project: which model family?

*Which situation is yours?*

- **One animal (or a few well-separated ones), real-time matters, no detector available:**
  use AnimalRTPose. `create` defaults to it; `--scale N` is the fastest, `--scale X` the most
  accurate.
- **Instances overlap, or you already have reliable boxes:** use AnimalViTPose. It crops each
  instance from its box, so overlapping animals stop competing for the same pixels. Pass
  `--model AnimalViTPose`.
- **You want to compare both:** create two projects on the same dataset descriptor and run
  the same `train` / `val`. `metrics.json` is directly comparable because the split, the OKS
  sigmas and the decoder are the same.

```{important}
The model family is fixed per project: it is written into `project.yaml` at `create` time
and selects the matching profile under `training.models`. Changing it means a new project,
not a configuration edit.
```

### Getting started with the top-down family

The top-down family needs boxes at every stage:

- training and validation read them from the split annotations;
- project prediction reads them the same way;
- live inference takes them from `--detector-weights`, or from an artifact whose output
  schema carries detections.

`animalpose-cli predict` and `animalpose-cli val` never load a detector on their own — the box
always comes from the labels or from the explicit detector argument. That is deliberate: it
keeps the evaluation honest and the deployment explicit.

## How to run AnimalPoseTracker

There are several ways to use it. Pick the one that fits your experience, and switch
whenever you like — they read and write the same files.

- **Option 1**: the [single-animal quick guide](quick-start/single_animal_quick_guide.md) — a full run on a built-in dataset.
- **Option 2**: the [desktop applications](gui/PROJECT_GUI.md) — the workspace, the
  annotator and the inferencer.
- **Option 3**: the command line — the most direct route to every stage, and the one the
  user guide follows.

### Option 1: the usage demo

One run end to end on the Tri-Mouse profile: `create`, `train`, `val`, `predict`, `export`,
then `infer` on the artifact. It writes every file the later stages expect, so it doubles as
a reference for what a project should look like.

### Option 2: using the desktop applications

```{warning}
On Windows, the first launch may ask for firewall access when a camera is opened in the
inferencer. Allow it on private networks; the applications never listen for inbound
connections.
```

The workspace covers project management, frame extraction, training, evaluation, prediction,
export and the 3D tab. The annotator draws boxes, classes, keypoints and the skeleton. The
inferencer runs an exported artifact on a camera, a video or a stream.

### Option 3: using the terminal

```bash
animalpose-cli --help
animalpose-cli create --dataset trimouse --workspace projects --name demo --worker me \
    --model AnimalRTPose --scale N --date demo
animalpose-cli train --config projects/demo-me-AnimalRTPose-N-demo/configs/runtime.yaml
```

Every command takes `--help`, and command-line values override `configs/runtime.yaml` for
that run only — nothing is written back.

## Useful links

- [Installing AnimalPoseTracker](installation.md) — environment, extras, engines.
- [User guide](main-workflows/user-guide.md) — the workflow in five phases.
- [Multi-camera 3D](Overviewof3D.md) — the 3D stage.
- [Deployment formats](recipes/OpenVINO.md) — the export matrix.
- [Citation](citation.md) — what to cite for each part.
