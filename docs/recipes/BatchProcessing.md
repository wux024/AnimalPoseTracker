# Batch and multi-camera runs

## Sources

- `--source` takes a directory, a glob, or a `.txt` manifest of paths
- a manifest is one path per line; blank lines are skipped
- each source writes into its own output directory when `--output-dir` is given

## Per-camera runs

- run `predict` once per camera with `--output-dir results/predict/<camera>`
- `pose3d` then reads each camera's `predictions.json` by default
- `--views camera=path` overrides an individual camera's file

## Shell loop

- a `for` loop over cameras is the simplest batch
- keep the exit code: a failed camera should stop the sequence
- name outputs by camera, not by timestamp, so `pose3d` finds them
