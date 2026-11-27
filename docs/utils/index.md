# Utility scripts

`tools/` holds five scripts that support development and evaluation rather than the workflow.

## What ships in `tools/`

| Script | Does |
|---|---|
| `convert_legacy_checkpoint.py` | converts older checkpoint formats to the current one |
| `demo_clips.py` | regenerates the demonstration GIFs in `docs/assets/demos/` |
| `screenshots.py` | regenerates the window screenshots in `docs/assets/screenshots/` |
| `chimpact_tracking_eval.py` | scores ChimpACT tracking video by video (CLEAR-MOT MOTA/MOTP and IDF1) |
| `dlc_benchmark.py` | turns a `predict` run into a DeepLabCut benchmark submission payload |

## When to use them

- after a release that changes the checkpoint layout: convert, do not retrain;
- when the demo clips or the screenshots no longer match the interface;
- when a tracking run needs numbers, or a DeepLabCut submission needs data.

See [tools reference](tools.md) for the flags.
