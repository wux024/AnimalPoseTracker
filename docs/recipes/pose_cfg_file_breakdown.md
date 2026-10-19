# model.yaml breakdown

## Top-level keys

- `nc` — class count the head is built with
- `kpt_shape` — `[K, D]`, keypoints and channels
- `scales` — compound scaling per named size
- `pretrained_urls` — one checkpoint per scale, for the one-stage family

## AnimalRTPose keys

- `backbone` — CSPNeXt stack, stride per stage
- `head` — three-level pose head with upsampling and concatenation
- `pretrained_urls` — full pose checkpoints, not backbone-only

## AnimalViTPose keys

- `backbone` — one positional ViT row
- `head` — SimCC, with the input grid and binning ratio derived from `image_size`
- `model_variants` — per-variant shape and MAE pretraining URL

## Editing safely

- change a scale block before changing the graph
- run `create` on a scratch project to validate the file without training
- a reordered positional row is a different model, not an error
