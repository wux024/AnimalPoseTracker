# Top-down AnimalViTPose at a glance

The second model family crops each instance from a box, so overlapping animals stop
competing for the same pixels. This page is the shortest path through it; the shared parts of
the workflow are in the [User guide](../main-workflows/user-guide.md).

## When to use the top-down family

| | AnimalRTPose | AnimalViTPose |
|---|---|---|
| Stage | one | two (box, then crop) |
| Needs a detector | no | for live inference only |
| Boxes at train/val/predict | predicted internally | read from the split's annotations |
| Scales | N / S / M / L / X | Small / Base / Large / Huge |
| Input | the full frame | one padded crop per instance |
| Best at | real-time, separated animals | overlapping instances, box-accurate data |

Choose top-down when you already trust your boxes: the model then spends its capacity on
keypoints inside a tight crop rather than on separating instances.

## Workflow

### 1. Create the project

```bash
animalpose-cli create --dataset trimouse --workspace projects --name vit --worker me \
    --model AnimalViTPose --scale B --date demo
cd projects/vit-me-AnimalViTPose-B-demo
```

The model family is written into `project.yaml` at creation; it selects the profile under
`training.models` and cannot be changed by editing a runtime key.

### 2. Check the boxes

Every stage reads boxes from the split annotations:

```bash
animalpose-cli load .
```

The summary reports the instances per split. An image with no box is invisible to training —
for this family, an unlabelled animal is not "learned as background", it is simply skipped.

How the crop is built, and where to tune it:

| Setting | Meaning | Lives in |
|---|---|---|
| `bbox_padding` | padding around the box before resizing (default 1.25) | `configs/model.yaml` |
| `image_size` | the square the crop is resized to | `configs/runtime.yaml` (`train.models.AnimalViTPose.training`) |
| `simcc_split_ratio` | SimCC binning granularity | `configs/model.yaml` |

### 3. Train

```bash
animalpose-cli train --config configs/runtime.yaml --epochs 100 --batch-size 16 --device 0
```

`auto` picks AdamW for this family. Training reads the boxes from the labels and applies the
same crop-and-normalize path the deployment uses, so what you validate is what you serve.

### 4. Validate

```bash
animalpose-cli val --config configs/runtime.yaml \
    --weights results/train/weights/best.pt
```

Flip test-time augmentation is **off** by default here, so the reported AP matches the
shipped SimCC decoder. Set `validation.flip_tta: true` to enable it and expect higher numbers
than the deployment path produces.

### 5. Predict

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --split test
```

Project prediction takes each instance box from the selected split, crops it, runs the ViT and
maps the SimCC points back to the source frame. It never loads a detector — the box always
comes from the labels.

### 6. Compare with the one-stage family

Run both families on the same dataset descriptor and split, then compare `metrics.json`:

```bash
animalpose-cli val --config configs/runtime.yaml --weights results/train/weights/best.pt
```

The numbers are directly comparable: same split, same OKS sigmas, same evaluation code. What
differs is the box source — AnimalRTPose predicts its own, AnimalViTPose is handed the
annotated ones. On clean boxes the top-down family usually wins on accuracy; on deployment
data without boxes, the one-stage family is the only one that runs unaided.

### 7. Deploy with a detector

Live inference is the one place the top-down family needs boxes it does not have:

```bash
animalpose-cli infer --config infer.yaml --detector-weights detector.onnx
```

Without `--detector-weights`, the inferencer refuses a SimCC artifact and names the flag. With
it, the stream becomes: detector → box → crop → ViT → points mapped back.

Export the pose model first; the detector is any artifact that emits boxes:

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx --dynamic --opset 17
```

The sidecar records the SimCC head, so the inferencer knows which decoder to apply.
