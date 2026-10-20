# 1. Annotate a few frames and export labels

Label a short clip with the annotator and leave the result where training will find it. A
minute of footage is enough to see whether the labels are right — the one-stage family needs
less data than most people expect.

This page assumes the project exists already; [Create the project
first](../your_data.md#create-the-project-first) is the one command that makes it, and the
place to record the keypoint count the annotator will show you.

## Picking the frames worth labelling

Frames go into `datasets/images/<split>/` of your project (`train` to learn from, `val` to
score against). Uniform sampling is fine for a first pass; after that, label what the model
gets wrong, not what it already does well. Ten frames that cover the pose variety beat a
hundred frames of one posture.

## Boxes, classes and keypoints

```bash
animalpose-annotator
```

Pick the project's `configs/dataset.yaml` and the images directory. Per animal:

1. **Box** — drag a tight rectangle. Padding is added by the trainer, not by you.
2. **Class** — pick from the list the dataset descriptor defined. A box with no class is not
   written; the window refuses to save an instance the dataset cannot describe.
3. **Keypoints** — place them in the order the sidebar shows. The skeleton overlay draws
   itself from `configs/dataset.yaml` as you click.

The full window walkthrough is in [Annotator GUI](../../gui/annotator_GUI.md); this page is
about where the output lands.

## The YOLO and COCO layouts

The annotator writes whichever format the project is configured for, and can switch between
them on the same frames:

| Format | Layout |
|---|---|
| YOLO-Pose | one `.txt` per image beside it, under `datasets/labels/<split>/` |
| COCO | one `annotations/<split>.json` per split |

`create` detects the format from the dataset root (an `annotations/*.json` means COCO),
or you fix it with `--annotation-format`. Both describe the same thing; COCO is easier to
diff, YOLO easier to move around.

## Checking the labels

```bash
animalpose-cli load .
```

`load` re-reads the project and reports the keypoint count, the class list, the instances
per split and the detected annotation format. A label file whose keypoint count disagrees
with `kpt_shape` fails here rather than mid-training — run this after every labelling
session, not just the first.

## Where to go next

- [2. Train and validate from Python](02_train_and_validate.md) — the labels you just wrote,
  through a short run.
- [Labeling guide](../../beginner-guides/labeling.md) — the labelling craft in more detail.
