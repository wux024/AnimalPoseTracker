# Fine-tuning a pretrained checkpoint

Fine-tuning is the normal way to start a project: the model already knows what an animal looks
like, and your labels only have to teach it the landmarks.

## Start from pretrained

```bash
animalpose-cli train --config configs/runtime.yaml \
    --weights pretrained/best.pt --epochs 60 --freeze 10
```

Three things differ from a from-scratch run:

| Setting | From scratch | Fine-tune |
|---|---|---|
| `--weights` | omitted (profile pretraining) | a checkpoint |
| `--freeze` | 0 | 5–10 |
| `--learning-rate` | profile default | a tenth of it |

The learning rate matters most: a full-rate schedule on a pretrained backbone destroys what it
learned in the first epoch.

## Freeze layers

`--freeze N` freezes the backbone for the first N epochs, then unfreezes everything. What it
buys:

- the head adapts to your keypoint layout without disturbing the features;
- a small dataset trains without overfitting the backbone;
- a short run finishes sooner.

What it costs: if your species looks nothing like the pretraining data, a frozen backbone caps
the accuracy — unfreeze earlier, or start from scratch.

## Small-data settings

| Symptom | Setting |
|---|---|
| overfitting after a few epochs | more augmentation, lower learning rate, freeze longer |
| validation loss flat | raise the learning rate, unfreeze earlier |
| good train, bad val | the split leaks: check that `val` frames come from different videos |

A few hundred labelled frames per behaviour is a realistic starting point; the point of
fine-tuning is that you do not need thousands.

## Compare before and after

```bash
animalpose-cli val --config configs/runtime.yaml --weights before.pt --output-dir results/val-before
animalpose-cli val --config configs/runtime.yaml --weights results/train/weights/best.pt \
    --output-dir results/val-after
```

Same split, same sigmas, same decoder — so the difference is the fine-tuning, not the setup. If
the number did not move, the labels did not teach anything new: look at the per-keypoint
breakdown before adding more epochs.

## Export

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx --dynamic --opset 17
```

A fine-tuned checkpoint exports exactly like a from-scratch one; the sidecar records the
preprocessing and the class/keypoint contract either way.
