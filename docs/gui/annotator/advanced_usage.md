# Annotator: classes, skeleton and multi-instance

## Class list

The class list comes from `configs/dataset.yaml` (`names`). It is not edited in the window —
change the descriptor and reload, so the labels and the model always agree on the class
index.

A single-species project usually keeps one class and lets the instance box do the separating.
Multiple classes are for genuinely different animals in the same recording.

## Skeleton editing

The skeleton is a list of keypoint pairs in the dataset descriptor:

```yaml
skeleton: [[0, 1], [0, 2], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
```

It is drawn, not labelled — every instance of the class shares it. If the drawing shows a line
between the wrong joints, the descriptor is wrong, not the annotation.

The keypoint **order** matters more than the names: the model learns position in the vector.
Reordering `keypoints_name` without reordering the labels shifts every point.

## Multiple instances per frame

Every box is one instance with its own set of keypoints. Two rules keep this usable:

- **draw the boxes first**, then place keypoints inside each — the active instance is the one
  whose box contains the cursor;
- **do not share keypoints between boxes**; a landmark belongs to exactly one instance.

For the one-stage family this is all the identity information there is. For the top-down family
each box becomes one crop, so a box that contains two animals teaches the model a crop that
contains two animals.

## Visibility

Some datasets carry a third keypoint channel (`kpt_shape: [K, 3]`). When it is present:

| Value | Meaning |
|---|---|
| visible | the landmark is in the frame and correctly placed |
| occluded | the animal is there, the landmark is hidden |
| absent | the landmark is outside the frame |

Validation skips absent points rather than scoring them as misses, which is why marking an
occluded point beats guessing one.
