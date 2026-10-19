# Labeling poses

Labels decide what the model can learn. This page is the short version; the
[Annotator GUI](../gui/annotator_GUI.md) covers the window in detail.

## The three things a label carries

- **A box** — one per instance, per frame. It is the instance boundary the top-down family
  crops from, and a training target for the one-stage family.
- **A class** — which kind of animal this box is. One class is enough for a single-species
  project.
- **Keypoints** — the landmarks, placed inside the box, in the order `configs/dataset.yaml`
  declares.

The skeleton is not drawn per label: it comes from the dataset descriptor and is shared by
every instance of that class.

## Consistency beats precision

The model learns the labelling convention you use, not the convention you intended:

- place a keypoint on the same physical landmark every time;
- if it is occluded, mark it invisible rather than guessing a position;
- keep the box tight; padding is added at training time, not at label time.

Inconsistent keypoints across frames teach the model an averaging of your habits, and no
amount of training fixes it.

## Saving

The annotator writes whichever format the project is configured for:

- **YOLO-Pose TXT** — one file per image, beside the image;
- **COCO Keypoints JSON** — one file per split.

Both are read directly. There is no export step before training.

## Checking your work

`animalpose-cli load .` prints the keypoint count and the instances per split. It catches the
mistakes that matter — a wrong `kpt_shape`, a label in the wrong folder, an annotation format
the project cannot read — before a training run wastes an hour on them.
