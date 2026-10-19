# Contributing to AnimalPoseTracker

Bug reports, dataset descriptors, model profiles, documentation fixes and code changes are all
useful. This page covers how to get a change to the point where it can be merged.

## Ways to contribute

- **Report a bug** — a command, the config and the traceback. Without the config it is usually
  not reproducible.
- **Add a dataset descriptor** — a YAML under `animalposetracker/cfg/datasets/` describing
  keypoints, skeleton, OKS sigmas and split folders. These are cheap to add and immediately
  useful.
- **Add a model profile** — a backbone or head that builds from the existing registry.
- **Fix documentation** — a page that is wrong is worse than a page that is missing.
- **Send code** — a new tracker, export format, metric or stage.

## Development setup

```bash
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[training,gui]"
```

For export or 3D work, add the extra you need (`.[export]`, `.[export-openvino]`,
`.[pose3d-numba]`, …). The list is in [Installing AnimalPoseTracker](docs/installation.md).

## Running tests

```bash
python -m pytest -q
```

The suite is organised by package; see the test table in the
[developer guide](docs/development/index.md). A change should come with the test file that
covers it — for a new CLI surface, `tests/test_cli.py`; for a 3D change, the matching
`tests/test_pose3d_*.py`.

## Documentation

The book lives in `docs/` and is built with Jupyter Book:

```bash
jupyter-book build .
```

`_config.yml` sets `only_build_toc_files: true`:

- a page must be listed in `_toc.yml` to be built;
- a `_toc.yml` entry pointing at a missing file fails the build.

So add the page first, then the entry. Links are relative to the page's own directory — after
moving a file, update the links that pointed at it.

## Pull requests

1. Branch from `main`.
2. Keep the change to one subject; a bug fix and a refactor in one PR are hard to review.
3. Add or update tests, and run the suite.
4. If behaviour changes, update the page that documents it — the docs are part of the change,
   not a follow-up.
5. Describe what you ran: the command, the dataset, the result.

## Reporting issues

Include:

- the `animalpose-cli` command or the GUI action;
- the project's `configs/runtime.yaml` (redact paths if needed);
- the traceback, or the console output around the failure;
- the environment: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## Code style

The repository follows the conventions already in the files: type annotations on public
functions, docstrings that say what a function returns, and comments that explain a decision
rather than restating the line. There is no formatter configured; match the surrounding code.
