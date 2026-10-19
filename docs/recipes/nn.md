# Editing model graphs

## The row grammar

- every row is `[from, repeats, module, args]`
- `from` is an index, `-1`, or a list of indices to concatenate
- `module` is a name the registry resolves
- `args` are positional; keyword options go in a trailing dict

## Scale variants

- the `scales` block holds `[depth, width, max_channels]` per named size
- the trainer maps `--scale` onto the block; the graph stays one file
- `model_variants` in the YAML carries architecture choices per variant

## What the builder does not do

- it does not infer shapes; a wrong channel count fails at build time
- it does not import torch in `spec.parse`; validation works without PyTorch
- it does not accept a module that is not registered
