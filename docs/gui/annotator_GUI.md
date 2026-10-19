# Annotator GUI

`animalpose-annotator` is the labeling window: it draws boxes, assigns classes, places
keypoints and records the skeleton, then saves labels in the format the project is configured
for.

## What the window is for

- Label a handful of frames per behaviour — the one-stage family needs less data than most
  people expect.
- Fix individual frames the model gets wrong, without leaving the project.
- Switch between YOLO-Pose and COCO output on the same set of frames.

It does not train, predict or export models; the workspace and the CLI do that.

## Table of contents

| Page | Covers |
|---|---|
| [Basic usage](annotator/basic_usage.md) | launch, load images, draw a box, place keypoints, save |
| [Classes, skeleton and multi-instance](annotator/advanced_usage.md) | class list, skeleton editing, several animals per frame |
| [Reviewing and exporting labels](annotator/review_and_export.md) | the review loop, corrections, YOLO and COCO export |
