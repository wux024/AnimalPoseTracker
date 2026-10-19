# PyTorch guides

AnimalPoseTracker is a PyTorch project end to end: both model families are defined as YAML
graphs, built by the same registry, trained by the same loop. These pages cover that layer for
people who want to change the model rather than just run it.

## What is here

| Page | Covers |
|---|---|
| [Training with PyTorch](user_guide.md) | the trainer, optimizers, EMA and AMP, freezing, resuming, distributed runs |
| [Configuration file reference](pytorch_config.md) | every registered key, generated from the schema |
| [Model architectures](architectures.md) | AnimalRTPose, AnimalViTPose, and the YAML graph language |
| [Benchmarking and evaluation](benchmarking.md) | what the metrics mean and how to reproduce a row |

## The two families

- **AnimalRTPose** — a CSPNeXt backbone with a one-stage pose head; boxes and keypoints come
  out of one forward pass.
- **AnimalViTPose** — a ViT backbone with a SimCC head, run on a crop supplied by a box.

Both are described in `configs/model.yaml` and both are built by
`animalposetracker.nn.builder`, so adding a backbone or a head is a YAML change plus a
registry entry, not a new training stack.

## Reading order

Start with [Model architectures](architectures.md) if you want to swap a backbone. Start with
[Training with PyTorch](user_guide.md) if you want to change how the model is trained. The
[configuration reference](pytorch_config.md) is for looking up a key, not for reading end to
end.
