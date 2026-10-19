# Model architectures

Both families are described as YAML graphs and built by one registry. This page explains what
the graphs contain, how the builder reads them, and what is deliberately not implemented.

## Introduction

A model file (`animalposetracker/cfg/models/*.yaml`) carries four things:

| Key | Meaning |
|---|---|
| `nc` | class count the head is built with |
| `kpt_shape` | `[K, D]` — keypoints and their channels |
| `scales` | compound scaling constants for the named sizes |
| `backbone` / `head` | the module graph, as a list of rows |

`configs/model.yaml` in a project starts as a copy of the selected profile, so editing the
architecture is editing that file — no Python required for a new depth, width or head channel
count.

## AnimalRTPose

One stage: a CSPNeXt backbone feeds a three-level pose head that predicts boxes and keypoints
together. There is no separate detector and no instance crop.

```yaml
backbone:
  - [-1, 1, STEM, [64]]                                  # 0  P1/2
  - [-1, 1, Conv, [128, 3, 2, 1]]                        # 1  P2/4
  - [-1, 3, CSPNeXtBlock, [128, True, False, True]]      # 2
  - [-1, 1, Conv, [256, 3, 2, 1]]                        # 3  P3/8
  - [-1, 6, CSPNeXtBlock, [256, True, False, True]]      # 4
  - [-1, 1, Conv, [512, 3, 2, 1]]                        # 5  P4/16
  - [-1, 6, CSPNeXtBlock, [512, True, False, True]]      # 6
  - [-1, 1, Conv, [1024, 3, 2, 1]]                       # 7  P5/32
  - [-1, 3, CSPNeXtBlock, [1024, False, True, True]]     # 8
```

The head upsamples, concatenates with the backbone's P4 and P3 outputs, and emits a pose
prediction at three strides. `YOLOPoseHead` is the module that turns those feature maps into
boxes plus keypoints.

The five scales are compound scaling constants:

| Scale | depth | width | max channels |
|---|---|---|---|
| N | 0.33 | 0.25 | 1024 |
| S | 0.33 | 0.50 | 1024 |
| M | 0.67 | 0.75 | 768 |
| L | 1.00 | 1.00 | 512 |
| X | 1.00 | 1.25 | 512 |

`pretrained_urls` in the same file points at full pose-model checkpoints, one per scale. They
are downloaded into `pretrained/` the first time a run needs them.

## AnimalViTPose

Two layers, and nothing else: a ViT emits one feature map, and a SimCC head consumes it
directly. There is no FPN and no multi-scale head — a top-down model sees one crop at one size.

```yaml
backbone:
  - [-1, 1, ViT, [384, 12, 12, 1536, 16, 256, {padding: 2}, {drop_path_rate: 0.1}]]
head:
  - [[-1], 1, SimCCHead, [num_keypoints, [256, 256], [16, 16], 2.0, [256], [4]]]
```

The ViT row is positional, not keyword-based: embed dims, layers, heads, feedforward channels,
patch size, image size, patch options, ViT options. The order is part of the grammar, so a
reordered row is a different model, not an error the parser can catch.

`model_variants` carries the Small/Base/Large/Huge shape and the MAE pretraining URL for each.
The trainer maps the project's `--scale` and `--image-size` onto the right variant, so one
graph covers all four.

The head's two grids — input resolution and the SimCC binning ratio — are derived from
`image_size` at build time. Changing the input size changes the head, which is why the exported
sidecar records it.

## The YAML graph language

Every row is `[from, repeats, module, args]`:

| Field | Meaning |
|---|---|
| `from` | index of an earlier row, `-1` for the previous row, or a list of indices to concatenate |
| `repeats` | how many times the module is stacked |
| `module` | a name resolved by the registry |
| `args` | positional arguments passed to the module |

Examples from the graphs above:

- `[-1, 3, CSPNeXtBlock, [...]]` — three blocks in sequence, each taking the previous output.
- `[[-1, 6], 1, Concat, [1]]` — concatenate the previous row with row 6 along dimension 1.
- `[[-1], 1, SimCCHead, [...]]` — a single-input head; the list form is required when the
  module is not a plain sequential block.

## Registry rules

`animalposetracker.nn.registry` maps a module name to a class. Three rules matter when adding
one:

1. **The name in the YAML is the class name** unless the registry declares an alias.
2. **Positional arguments are passed through unchanged** — a module that wants keyword options
   takes them as a trailing dict in the row (`{padding: 2}`).
3. **The builder stays torch-free until build time** — `nn.spec` parses the graph without
   importing torch, which is what lets `create` validate a model file on a machine that has no
   PyTorch installed.

## What is not implemented

RTMO appears in the codebase as an assignment and loss port — the matching and the loss
functions exist, and the decoder layer knows the `mmpose_rtmo_mmdeploy` output schema. It is
**not** a trainable family: there is no RTMO model file, no backbone registration, and the
inferencer rejects the schema. Do not select it in a project.

Multi-animal association (PAF assembly, identity heads) is likewise absent: the one-stage head
predicts a fixed set of instances per image, and identity across frames is recovered by the
tracker, not by the model.
