# Developer guide

How the repository is organised, where a change belongs, and how to check that it did not
break anything.

## Repository layout

| Path | Responsibility |
|---|---|
| `animalposetracker/cli.py` | the `animalpose-cli` entry point and subcommand dispatch |
| `animalposetracker/project/` | create, load, project files, weights |
| `animalposetracker/config/` | schema registry, resolver, writer, generated docs |
| `animalposetracker/nn/` | the YAML graph language, module registry, builder, heads |
| `animalposetracker/data/` | datasets, augmentation, COCO and YOLO readers, top-down crop |
| `animalposetracker/training/` | training loop, optimizer, EMA, checkpoints, profiles |
| `animalposetracker/evaluation/` | OKS/PCK/AUC/EPE metrics and the per-family adapters |
| `animalposetracker/prediction/` | project prediction and rendering |
| `animalposetracker/inference/` | the deployment-side engine, session, schema resolution |
| `animalposetracker/export/` | the format matrix and per-backend exporters |
| `animalposetracker/postprocessing/` | output decoders, filters, tracking glue |
| `animalposetracker/tracking/` | the six trackers |
| `animalposetracker/pose3d/` | calibration, association, triangulation, rendering |
| `animalposetracker/gui/` | the three PySide6 windows |
| `animalposetracker/cfg/` | package defaults, model profiles, dataset descriptors |

A change belongs in the package that owns the noun: a new tracker in `tracking/`, a new export
format in `export/`, a new metric in `evaluation/`.

## Architecture

The data flow is one direction:

```text
config (schema + resolver)
      │
      ├── data ──► training ──► checkpoint ──► evaluation
      │                              │
      └── model (YAML graph) ────────┴──► prediction ──► records ──► tracking/filter
                                                    │
                                                    └──► pose3d ──► render
```

Two invariants keep it honest:

- **one record schema** — `predictions.json` is the same file the 3D stage reads, the tracker
  reads, and the inferencer writes;
- **one decoder** — training, validation, prediction and deployment all call
  `postprocessing.output_decoders`, so a metric describes what ships.

## Configuration registry

To add a key:

1. register it in `animalposetracker/config/schema.py` (name, scope, group, type, apply level,
   GUI visibility, one-line note, and the `views` it belongs to);
2. give it a default in `cfg/training.yaml` or `cfg/runtime_defaults.yaml`;
3. run `python -m pytest tests/test_config_schema.py` — it checks the two stay aligned;
4. regenerate the reference:
   `python -m animalposetracker.config.docs --write docs/pytorch/pytorch_config.md`.

The GUI's parameter trees and the generated reference are both driven by that annotation, so a
registered key shows up everywhere and nowhere else.

## Models

`animalposetracker/nn/spec.py` parses a YAML graph without importing torch; `nn/builder.py`
builds it. The split matters: `create` validates a model file on machines without PyTorch.

Adding a module means registering its name in `nn/registry.py` and using that name in a YAML
row. See [Model architectures](../pytorch/architectures.md) for the row grammar and the
positional-argument rules.

## Tests

The suite is organised by package; a change should come with the file that covers it:

| Area | Test files |
|---|---|
| CLI and project | `test_cli.py`, `test_config_resolver.py`, `test_config_schema.py` |
| data and labels | `test_label_conversion.py`, `test_shape_collection.py`, `test_batch_probe.py` |
| models and training | `test_animalvitpose_training.py`, `test_nms_core.py` |
| inference | `test_inference_session.py`, `test_inference_config.py`, `test_stream_inference.py`, `test_schema_resolution.py`, `test_detector_adapter.py` |
| prediction and rendering | `test_pose_processing.py`, `test_rendering.py`, `test_tracking.py` |
| 3D | `test_pose3d_accelerate.py`, `test_pose3d_associate.py`, `test_pose3d_bundle.py`, `test_pose3d_calibrate.py`, `test_pose3d_export.py`, `test_pose3d_gui_bridge.py`, `test_pose3d_protocol.py`, `test_pose3d_video.py` |
| GUI | `test_gui.py`, `test_review_fixes.py` |

```bash
python -m pytest -q                      # the whole suite
python -m pytest tests/test_pose3d_protocol.py -q
```

## CI and docs build

`.github/workflows/docs.yml` builds this book on every push and deploys it on manual dispatch.
For a local build:

```bash
jupyter-book build .
```

The config sets `only_build_toc_files: true`, so a page that is not referenced from `_toc.yml`
is not built — and a `_toc.yml` entry pointing at a missing file fails the build. Add the page
first, then the TOC entry.

## Releasing

1. bump `__version__` in `animalposetracker/__init__.py`;
2. update `## Major Versions` and `# News` in the README;
3. tag the release (`vX.Y.Z`), and attach the checkpoint-free source tarball;
4. the docs workflow deploys the book on the next manual dispatch.

There is no changelog file: the README's version section and the git history are the record.
