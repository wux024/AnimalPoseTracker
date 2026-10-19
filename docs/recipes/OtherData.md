# Bringing your own dataset

A dataset is described by one YAML. This recipe shows the keys, the two label layouts, and the
two ways to point `create` at it.

## Descriptor keys

```yaml
path: /data/mouse              # root; absolute or relative to the project
train: images/train
val: images/val
test: images/test
kpt_shape: [12, 3]             # keypoints, channels (2 = x,y; 3 = x,y,visible)
names:
  0: mus1
  1: mus2
skeleton: [[0, 1], [0, 2], [3, 4], [4, 5]]
flip_idx: [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11]
kpt_oks_sigmas: [0.026, 0.025, 0.025, 0.035]
```

| Key | Required | Meaning |
|---|---|---|
| `path` | yes | dataset root |
| `train` / `val` / `test` | yes | split folders, relative to `path` |
| `kpt_shape` | yes | keypoint count and channels |
| `names` | yes | class index → name; the count is the class count |
| `skeleton` | recommended | keypoint pairs to draw and evaluate |
| `flip_idx` | recommended | left/right pairs, for flip augmentation |
| `kpt_oks_sigmas` | recommended | per-keypoint tolerance for AP |
| `annotation_format` | optional | `auto` (default), `coco`, `yolo` |

## YOLO layout

```text
datasets/
├── images/train/   frame_0001.jpg
├── images/val/
├── images/test/
├── labels/train/   frame_0001.txt
├── labels/val/
└── labels/test/
```

Each `.txt` line: `class cx cy w h kx1 ky1 v1 kx2 ky2 v2 ...`, all normalized to the image.

## COCO layout

```text
datasets/
├── images/train/
├── images/val/
├── images/test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

`person_keypoints_{split}2017.json` names are detected too. `train_annotations` and
`val_annotations` override the discovery when the files are named differently.

Images, boxes and keypoints are read straight from JSON; nothing converts the dataset.

## OKS sigmas and the flip table

Two keys decide whether your metric is meaningful and whether augmentation helps:

- **`kpt_oks_sigmas`** — one positive value per keypoint. Without it the trainer warns and falls
  back to a uniform `1 / K`, which makes AP incomparable with published numbers.
- **`flip_idx`** — maps each keypoint to its left/right counterpart. Flip augmentation uses it;
  a wrong table teaches the model that left and right are the same landmark.

## Register vs path

- **Register** a descriptor under `animalposetracker/cfg/datasets/` to pass it by name
  (`--dataset trimouse`). Useful when the layout is stable and shared.
- **Pass a path** (`--dataset /data/mouse/dataset.yaml`) for anything local or one-off.

Both are the same call; only the lookup differs.
