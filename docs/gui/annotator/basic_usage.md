# Annotator: basic usage

## Launch and load

```bash
animalpose-annotator
```

Pick the configuration file (the project's `configs/dataset.yaml`) and the images directory.
The window lists every image it finds and remembers where you stopped.

https://github.com/user-attachments/assets/7d820fbe-390e-4387-9678-ac229e930cda

## Navigate images

| Action | Shortcut |
|---|---|
| next / previous image | arrow keys |
| zoom | scroll wheel |
| pan | middle-drag |
| fit to window | the toolbar's fit button |

The frame counter and the file name are in the status bar — the fastest way to know whether
you are looking at the image you think you are.

## Draw a box

1. Pick the **box** tool.
2. Drag from one corner of the animal to the other. The box should be tight: padding is added
   by the trainer, not by the label.
3. Pick the class for the box. A single-class project can leave it at the default.

A box with no class is not written; the window refuses to save an instance that would decode
to something the dataset cannot describe.

## Place keypoints

1. Pick the **keypoint** tool.
2. Click the landmark the dataset expects, in order. The active keypoint is shown in the
   sidebar, so a click always lands on the one you mean.
3. For an occluded landmark, mark it invisible rather than guessing a position — a guessed
   point is a wrong supervision signal, an invisible one is skipped.

https://github.com/user-attachments/assets/a22b28df-4b12-48b0-a087-4ab07f6ea17f

## Save

Save writes the current image's labels in the format the project uses:

- YOLO-Pose TXT beside the image;
- COCO JSON for the split, rewritten with the new instance.

Saving is per image, so an interrupted session loses only the image that was open.
