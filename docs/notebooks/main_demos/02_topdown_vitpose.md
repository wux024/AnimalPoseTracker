# Top-down walkthrough with AnimalViTPose

The same Tri-Mouse data through the second model family: boxes come from the labels, each
instance is cropped, and a ViT decodes keypoints with SimCC heads. The point of this
walkthrough is to see where the box enters every stage — and what that buys you.

## Why boxes matter

AnimalRTPose separates instances and regresses keypoints in one pass over the full frame.
AnimalViTPose never sees the full frame: it receives one padded crop per box, so two
overlapping animals never compete for the same pixels. The trade is that something else has
to supply the box — during training and validation that is the annotation, during project
prediction it is the split's labels, and during live inference it is a detector you attach
yourself.

## Steps

### 1. Create the project with the second family

```bash
animalpose-cli create --dataset trimouse --dataset-root /path/to/trimouse \
    --workspace projects --name vit --worker me \
    --model AnimalViTPose --scale B --date demo
cd projects/vit-me-AnimalViTPose-B-demo
```

The family is written into `project.yaml` at creation. Scales run Small / Base / Large /
Huge (`S` / `B` / `L` / `H`).

### 2. Check the boxes before training

```bash
animalpose-cli load .
```

For this family an unlabelled animal is not "learned as background" — an image with no box
is skipped entirely. The crop is built from `configs/model.yaml`:

| Setting | Meaning | Lives in |
|---|---|---|
| `bbox_padding` | padding around the box before resizing (default 1.25) | `configs/model.yaml` |
| `image_size` | the square the crop is resized to | `configs/runtime.yaml` |
| `simcc_split_ratio` | SimCC binning granularity | `configs/model.yaml` |

### 3. Train

```bash
animalpose-cli train --config configs/runtime.yaml --epochs 100 --batch-size 16 --device 0
```

Training crops with the same padding and normalization the deployment path uses, so what
you validate is what you serve. `auto` picks AdamW for this family.

### 4. Validate — and know what flip TTA does

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt
```

`results/val/metrics.json` reports AP, AR, PCK, AUC and EPE. Flip test-time augmentation is
off by default, so the number you see matches the shipped SimCC decoder. Setting
`validation.flip_tta: true` averages each crop with its mirror and usually raises AP — but
the deployment path does not do this, so the TTA number is not what your exported model
produces.

### 5. Predict on a split

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --split test
```

The worker takes each instance box from the split's annotations, crops, runs the ViT, and
maps the SimCC coordinates back into the source frame. It never loads a detector.

### 6. Compare with the one-stage family

Train AnimalRTPose on the same descriptor and split (walkthrough
[01](01_quickstart_cli.md)), then put the two `metrics.json` files side by side. Same split,
same OKS sigmas, same evaluation code — the only difference is the box source. On clean
annotated boxes the top-down family usually wins on accuracy; on deployment data without
boxes, the one-stage family is the only one that runs unaided.

## Where to go next

- [Top-down at a glance](../../quick-start/topdown_quick_guide.md) — the short version.
- [Live inference](../extra/02_live_stream.md) — attaching a detector so this family runs on
  a stream.
