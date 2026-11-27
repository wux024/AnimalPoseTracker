# Bringing your own dataset

A dataset is described by one YAML. This recipe shows the keys, the two label layouts, and the
three ways to point `create` at your data — a packaged name, your own YAML, or the layout flags
`--keypoints`/`--classes` with no file at all.

Descriptors for 27 public datasets ship under `animalposetracker/cfg/datasets/`; each one
opens with its source and download link. The toolbox never downloads dataset images.

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
| `annotation_format` | optional | `auto` (default), `coco`, `yolo`. `create` resolves `auto` from the root (an `annotations/*.json` means `coco`) and writes the answer in; the loader accepts all three and falls back to `coco` when the key is missing |

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

YOLO labels feed validation exactly like COCO JSON does — no `annotations/` folder is
needed. A split with nothing to train or validate on (no images, or no annotated
instances) raises instead of silently training without validation.

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

One fixed location: the loader reads `<path>/annotations/<split>.json` and nothing else —
no name patterns to match, no override keys. With `annotation_format: coco` a missing JSON
is an error; with `auto` a missing JSON falls through to the YOLO layout.

Images, boxes and keypoints are read straight from JSON; nothing converts the dataset. Each
`images[].file_name` is resolved in one of two ways: a bare name (what the annotator writes)
joins the split's image folder from the `train:`/`val:`/`test:` key, and a name with folder
parts (what external COCO files carry) is relative to the dataset root. Absolute paths are
refused.

## OKS sigmas and the flip table

Two keys decide whether your metric is meaningful and whether augmentation helps:

- **`kpt_oks_sigmas`** — one positive value per keypoint. Without it the trainer warns and falls
  back to a uniform `1 / K`, which makes AP incomparable with published numbers.
- **`flip_idx`** — maps each keypoint to its left/right counterpart. Flip augmentation uses it;
  a wrong table teaches the model that left and right are the same landmark. Leave it out and
  training keeps horizontal flips off, so nothing is mirrored without swapping labels.

## Three ways to describe the layout

- **Register** a descriptor under `animalposetracker/cfg/datasets/` to pass it by name
  (`--dataset trimouse`). Useful when the layout is stable and shared.
- **Pass a path** (`--dataset /data/mouse/dataset.yaml`) for anything local or one-off.
- **Describe it inline** (`--keypoints 12 --classes 2 --class-names pup,mus`) when there is no
  file to keep: `--keypoint-names` and `--skeleton 0-1,1-2` are optional. A `flip_idx` table has
  no inline form — keep one in a file when you want flip augmentation.

All three end in the same `configs/dataset.yaml`; only the source differs.
