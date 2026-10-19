# Managing a project

The **Manage** tab covers everything between creating a project and training on it: the
files, the frames, the splits and the configuration.

## Project files

| File | What it holds |
|---|---|
| `project.yaml` | identity: name, worker, model, scale, date, flags |
| `configs/dataset.yaml` | keypoints, classes, skeleton, OKS sigmas, split folders |
| `configs/model.yaml` | backbone, head, scale, preprocessing |
| `configs/runtime.yaml` | every runtime parameter, by flow |
| `configs/cameras.yaml` | multi-camera calibration, 3D projects only |

Editing a file outside the window is fine — the tab re-reads it when you switch back, and
reports a key it cannot parse rather than silently keeping the old value.

## Sources and frame extraction

Frames come from videos in `sources/`. The tab offers two extraction modes:

- **Automatic** — samples the video uniformly; set the count and it picks evenly spaced
  frames. Best when the behaviour is spread throughout the recording.
- **Manual** — scrub the video and capture the frames you want. Best when the interesting
  moments are rare.

Extracted frames land in `sources/extracted/` with the source name in the filename, so a frame
can always be traced back to the video and timestamp it came from.

## Dataset splits

The three splits — `train`, `val`, `test` — are directories under `datasets/images/`. The tab
shows how many frames and how many labels each split holds, which is the fastest way to catch
a label that landed in the wrong folder.

| Split | Used for |
|---|---|
| `train` | the epochs |
| `val` | the per-epoch metric and early stopping |
| `test` | the final number, and `predict --split test` |

A frame without a label is skipped by training, not treated as background. If the counts
disagree, fix the labels before starting a run.

## Editing configuration

The tab exposes the common runtime values directly: epochs, batch size, image size, device,
workers. Anything not on the form is in `configs/runtime.yaml` and can be edited by hand.

The full key list, with types and defaults, is the
[configuration reference](../pytorch/pytorch_config.md) — generated from the schema, so it
cannot drift from what the code accepts.
