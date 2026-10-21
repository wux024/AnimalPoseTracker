# Reviewing and exporting labels

## The review loop

Label a few frames, train briefly, predict on the same frames, and look at the difference. The
frames where the skeleton is visibly off are worth relabelling — that is the loop that raises
accuracy faster than labelling more of the same.

The practical order:

1. label 20–40 frames across the behaviours you care about;
2. train 50 epochs and run `val`;
3. predict on the labelled frames and render the overlay;
4. relabel the frames where the overlay disagrees with the animal;
5. retrain with `--pretrained best.pt` and `--freeze` for the first epochs.

## Correcting labels

Open the same image in the annotator and drag the offending keypoint. The save replaces that
image's instance list, so a correction is final — there is no undo history across saves.

When a box is wrong rather than a keypoint, delete the instance and draw it again. Editing a
box after placing keypoints can leave points outside it, which the top-down crop will then cut
off.

## YOLO export

The project writes YOLO-Pose TXT directly: one line per instance, class followed by the
normalized box and the keypoint coordinates. No export command is needed — the files in
`labels/<split>/` are the format.

## COCO export

COCO is written per split, under `annotations/`. The same instances are written with absolute
pixel coordinates and a category id from the class list.

Both formats are read by the trainer; the choice is a convenience, not a conversion step.

## Consistency checks

Before a long run, check the three things that actually break:

| Check | How |
|---|---|
| keypoint count | `animalpose-cli load .` reports `kpt_shape` against the labels |
| split placement | the Manage tab counts frames and labels per split |
| class ids | a label with a class the descriptor does not define fails at `load` |

The window and the CLI read the same files, so a problem found in one is visible in the other.
