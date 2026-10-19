# Utility scripts

`tools/` holds three scripts that support development rather than the workflow.

## What ships in `tools/`

| Script | Does |
|---|---|
| `convert_legacy_checkpoint.py` | converts older checkpoint formats to the current one |
| `demo_clips.py` | regenerates the demonstration GIFs in `docs/assets/demos/` |
| `screenshots.py` | regenerates the window screenshots in `docs/assets/screenshots/` |

## When to use them

- after a release that changes the checkpoint layout: convert, do not retrain;
- when the demo clips or the screenshots no longer match the interface.

See [tools reference](tools.md) for the flags.
