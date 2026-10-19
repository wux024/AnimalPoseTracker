# Installing AnimalPoseTracker

AnimalPoseTracker installs from source: clone the repository and install it in editable mode
with the extras you need. There is no compiled step, and nothing outside PyPI is required
except PyTorch, which comes from its own index.

## Using Conda

### 🚨 Before you start...

- **Python 3.10 or newer.** The package targets 3.10+ and the CI runs 3.10.
- **A few GB of free disk** for the environment and checkpoints; datasets add on top.
- **A CUDA GPU is optional.** Training runs on the CPU, slowly; inference on an exported
  artifact needs no PyTorch at all.
- **Pick the extras you need now.** They are independent and can be installed later, but
  installing them up front avoids a second resolution pass.

### Step 1: create the environment

```bash
conda create -n animalposetracker python=3.10
conda activate animalposetracker
```

`venv` works the same way if you prefer it:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### Step 2: install PyTorch

Pick the line that matches the machine (the full matrix is at
<https://pytorch.org/get-started/locally/>):

```bash
# CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

Skip this step if you only run exported artifacts — `animalpose-cli infer` needs no PyTorch.

### Step 3: install the package

```bash
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
pip install -v -e ".[training,gui]"    # desktop: GUI + built-in training
pip install -v -e ".[training]"        # headless server: CLI + built-in training
```

The `-e` (editable) install points the environment at the checkout, so source edits take
effect without reinstalling. `-v` prints the dependency resolution, which is useful the
first time.

### Step 4: check the install

```bash
animalpose-cli --version
animalpose-cli --help
python -c "import torch; print(torch.cuda.is_available())"
```

The last line prints `True` when the PyTorch build can see a CUDA device. `False` is
expected on a CPU-only machine; training then falls back to the CPU, and `--device cpu` says
so explicitly.

### Conda environment management tips

```bash
conda env export > environment.yml     # record the exact set
conda env remove -n animalposetracker  # start over
```

Two projects can share one environment; the package is installed once and every project
directory is data, not code.

## Other ways to install

### git clone

The four steps above are the git-clone route: clone, then `pip install -e` with the extras
you need.

### `uv` (recommended for developers)

`uv` resolves the same extras from `pyproject.toml` and is faster for development:

```bash
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
uv sync --extra training --extra gui
uv run animalpose-cli --version
```

### `pip`

```bash
pip install "animalposetracker[training,gui]"
pip install "animalposetracker[training]"      # headless
```

### Docker

```bash
docker build -t animalposetracker .
docker run --gpus all -it --rm -v "$PWD:/work" animalposetracker
```

The repository does not ship a Dockerfile yet; [Running in containers](docker.md) has a
recipe you can paste into one.

### Creating your own environment

Any environment that meets the requirements works; conda is a convenience, not a
dependency. What matters is the Python version and, for training, a PyTorch build that
matches the machine.

The `uv`, `pip` and Docker routes assume the PyPI release and a repository Dockerfile; the
git-clone route above is the one that works today.

## Updating

```bash
git pull
pip install -v -e ".[training,gui]"
```

An editable install picks up source changes on its own. Re-run the install when
`pyproject.toml` changes, so new dependencies are pulled in — a new extra, a raised pin, or
a new entry point.

### Data compatibility

Configuration keys and the project layout are versioned together with the package. A project
created by an older release may carry keys that no longer exist, or miss keys the new
release expects; the resolver reports the mismatch by name rather than guessing.

In that case, recreate the project with the current `create` and copy the dataset paths and
labels across. Checkpoints (`best.pt` / `last.pt`) stay loadable across minor versions, and
`tools/convert_legacy_checkpoint.py` converts the older formats.

## GPU Support

### General GPU support

Training and TensorRT export want an NVIDIA GPU with a CUDA build that matches the driver.
`nvidia-smi` reports the driver's CUDA version; install a PyTorch wheel at or below it.

```bash
nvidia-smi                      # driver and the CUDA version it supports
python -c "import torch; print(torch.version.cuda)"
```

Inference is more forgiving: ONNX Runtime, OpenVINO, CoreML, CANN and OpenCV each run on
their own hardware, and only the engine you deploy with has to be installed.

### Notes

- A CPU-only install trains and predicts correctly, just slower; nothing in the data path
  requires a GPU.
- Mixed precision (`--amp`) is a training flag, not an install step; it falls back cleanly on
  devices without tensor cores.
- Multi-GPU training uses `--device 0,1` and needs no extra package beyond PyTorch.

## Troubleshooting

### PyTorch

**`torch.cuda.is_available()` is False.** The wheel does not match the driver. Check
`nvidia-smi` for the CUDA version the driver supports, then reinstall from the matching index
URL above. CPU-only training still works with `--device cpu`.

**`CUDA out of memory` during training.** Lower `--batch-size`, lower `--image-size`, or
enable `--amp`. The error names the tensor shape that failed, which tells you which of the
three helped.

### Qt

**The GUI will not start on Linux.** Qt needs the system's X/Wayland libraries; install the
distribution packages for Qt 6 if the PySide6 wheel reports a missing `.so`.

**The GUI starts but the camera list is empty.** `PyCameraList` is a Windows-only extra; on
other platforms the inferencer accepts a video file or a stream URL instead.

### AnimalPoseTracker

**`animalpose-cli: command not found`.** The environment is not active, or the install
finished in a different one. `pip show animalposetracker` prints where it landed.

**Imports fail after `git pull`.** A dependency changed. Re-run the `pip install -e` line for
the extras you use.

**`create` reports a dataset descriptor it cannot resolve.** Pass a path to the YAML instead
of a registered name; the packaged names live in `animalposetracker/cfg/datasets/`.

## System-wide installation considerations

A system-wide install (`pip install` outside a virtual environment) works, but the package
pulls PyTorch, PySide6 and the export toolchain — installing those into the system Python
tends to break other tools.

Prefer one environment per project family, and keep the environment outside the project
directory: the project holds data and configuration, not the interpreter. On shared
machines, install as the user who runs the jobs, so the GUI and the CLI see the same
environment.

## Hardware considerations

The package runs wherever Python and PyTorch run. The numbers below are inference latency
measured on the machines in the README; they are a guide to the order of magnitude, not a
benchmark of your data.

| Machine | Compute unit | Inference |
|---|---|---|
| Mac Mini | integrated | 3.2 ms |
| Lenovo laptop | CPU only | 54.5 ms |
| Lenovo desktop | CPU only | 37.7 ms |
| Workstation 1 | RTX 4090 | 6.0 ms |
| Workstation 1 | Intel integrated GPU | 10.0 ms |
| Workstation 1 | Intel NPU | 25.5 ms |
| Workstation 1 | CPU only | 18.2 ms |
| Workstation 2 | RTX 4080 | 2.6 ms |
| Workstation 2 | CPU only | 14.2 ms |
| Jetson AGX Orin | Ampere GPU | 4.2 ms |
| Atlas 200I DK A2 | DAVINCI V300 | 15.6 ms |

For training, the GPU matters much more than for inference: a full run on a CPU is possible
but slow, and the one-stage models are the practical choice there. For deployment, an edge
device plus an exported artifact covers the cases where a workstation is not an option.
