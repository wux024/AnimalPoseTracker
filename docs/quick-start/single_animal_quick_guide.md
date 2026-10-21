# Single-animal at a glance

The shortest path from a dataset to a trained one-stage model. Every command assumes the
install is done ([Installing AnimalPoseTracker](../installation.md)) and runs from the
project directory. The full explanation of each step is in the
[User guide](../main-workflows/user-guide.md).

## Start the environment

```bash
conda activate animalposetracker
animalpose-cli --version
```

## Workflow

### 1. Create a project

```bash
animalpose-cli create --dataset trimouse --workspace projects --name demo --worker me \
    --model AnimalRTPose --scale N --date demo
cd projects/demo-me-AnimalRTPose-N-demo
```

`--dataset` takes a packaged name (`animalposetracker/cfg/datasets/`) or a path to your own
descriptor. The project directory is `name-worker-model-scale-date`.

### 2. Configure the project

Defaults are already usable. Change what matters for your data in `configs/runtime.yaml`:

```yaml
train:
  models:
    AnimalRTPose:
      training:
        epochs: 100
        batch_size: 8
        image_size: 640
```

`configs/dataset.yaml` already carries the keypoint names, the skeleton and the split
folders; edit it only when you bring your own dataset.

### 3. Extract frames

Frame extraction is a desktop step. Open the workspace and use the **Manage** tab:

```bash
animalposetracker
```

Automatic extraction samples the video — `kmeans` clustering by default (one representative per
appearance cluster), or `uniform` spacing; manual capture happens in the annotator. Frames land
in `sources/extracted/`. The CLI has no `extract` subcommand — copy the frames into
`datasets/images/<split>/` yourself if you prefer the terminal.

### 4. Label frames

```bash
animalpose-annotator
```

Draw a box per animal, then place the keypoints. The skeleton is drawn from
`configs/dataset.yaml`. Labels are written as YOLO-Pose TXT beside each image, or as COCO
JSON per split, whichever the project is configured for.

### 5. Check labels

```bash
animalpose-cli load .
```

`load` prints the keypoint count, the class list, the instances per split and the annotation
format it detected. A label file whose keypoint count disagrees with `kpt_shape` fails here
rather than during training.

### 6. Review the training inputs

There is no separate "build dataset" command: the dataloader reads the split folders on every
run. What you can check up front:

- every image has a label (or is deliberately unlabelled);
- `kpt_shape`, `flip_idx` and `kpt_oks_sigmas` describe your landmark set;
- the class names match what the annotator wrote.

### 7. Train the network

```bash
animalpose-cli train --config configs/runtime.yaml \
    --epochs 100 --batch-size 8 --device 0
```

Watch `results/train/training.jsonl` while it runs; `results/train/results.png` is written at
the end. Early stopping can end the run before `--epochs`, and that is a normal completion.

### 8. Evaluate the trained network

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt
```

`results/val/metrics.json` holds AP, AR, PCK, AUC and EPE, with the per-keypoint breakdown
where it applies.

### 9. Analyze videos

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/ --conf 0.25
```

`results/predict/predictions.json` records the keypoints; the rendered video lands beside it.

### 10. Filter predictions

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/ \
    --tracker bytetrack --pose-filter median --filter-window 7
```

Tracking adds identities across frames, filtering smooths each track. Both write sibling
files and leave `predictions.json` untouched.

### 11. Summarize results

The summaries that ship are the training curves, the validation metrics and — for 3D
projects — a trajectory plot:

```bash
animalpose-cli pose3d-render --config configs/runtime.yaml --plot results/pose3d/trajectory.png
```

For a 2D sequence there is no plotting command; read the records with the API and plot them
with your own tools.

### 12. Create labeled videos

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source videos/clip.mp4 \
    --show-skeletons --show-keypoints --line-width 3
```

The overlay is written into `results/predict/videos/`.

That is the full 2D loop. The refinement loop — add frames, retrain from `best.pt`, compare
on the same split — is Phase 5 of the [User guide](../main-workflows/user-guide.md).
