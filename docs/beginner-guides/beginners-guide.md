# Setting up a new project

The first ten minutes: install, launch, and create a project you can train on.

## Installation

Follow [Installing AnimalPoseTracker](../installation.md) with the `gui` extra. The window
will not start without PySide6:

```bash
pip install -v -e ".[training,gui]"
```

## Launching the workspace

```bash
conda activate animalposetracker
animalposetracker
```

The status bar reports the Python, PyTorch and CUDA versions the window found. If PyTorch is
missing, training tabs stay disabled while the annotator and the management tabs still work.

## Creating or loading a project

Two buttons, two situations:

- **Create** — start a new project from a dataset descriptor or your own images.
- **Load** — open an existing project directory; everything the stages wrote is already
  there.

A project directory is self-describing: `project.yaml` names it, `configs/` describes it. The
window never stores state of its own, so closing it and reopening the project loses nothing.

## New project step-by-step

1. **Pick the dataset.** A packaged descriptor (Tri-Mouse, AP-10K, AnimalPose, …) or a path to
   your own `dataset.yaml`. The descriptor supplies the keypoint names, the skeleton, the OKS
   sigmas and the split folders.
2. **Pick the model family.** AnimalRTPose for one-stage, AnimalViTPose for top-down. This is
   written into `project.yaml` and cannot be changed later.
3. **Pick the scale.** N/S/M/L/X for the one-stage family, Small/Base/Large/Huge for the
   top-down. Start small; scale up when the metrics justify it.
4. **Name it.** The directory becomes `name-worker-model-scale-date`.
5. **Choose where it lives.** A workspace folder that already holds other projects is fine.

The form only writes what you fill in; everything else keeps the model profile's defaults.

## Next steps

- [Managing a project](manage-project.md) — frames, splits, configuration.
- [Labeling poses](labeling.md) — the annotator.
- [Training and evaluation](Training-Evaluation.md) — the training form.
