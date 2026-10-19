# Demos on your own data

Three walkthroughs that start from your frames rather than a packaged dataset: label a few,
train on them, and take the model to new material.

## What these pages are

They follow one project from raw video to an exported artifact, with the commands you would
actually type. They are not notebooks — nothing here requires Jupyter, and every step runs in
a terminal or from Python.

## Before you start

- The package installed with the `training` extra, plus `gui` if you want the annotator;
- a short clip — a minute is enough to see whether the labels are right;
- a decision about keypoints: which landmarks you will track, in which order.

## The three walkthroughs

1. [Annotate a few frames and export labels](your_data/01_annotate_and_export.md) — the
   annotator, the two label formats, and the check before training.
2. [Train and validate from Python](your_data/02_train_and_validate.md) — a short run, the
   records it writes, and how to read the metrics.
3. [Predict and export](your_data/03_predict_and_export.md) — new material, tracking and
   filtering, then an ONNX artifact.

The same steps with full explanations are in the
[User guide](../main-workflows/user-guide.md).
