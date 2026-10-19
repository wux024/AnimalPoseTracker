# Running AnimalPoseTracker in containers

The repository does not ship a Dockerfile or a published image. This page is a recipe you
can paste into one: it covers the headless CLI, the GPU case, and what does **not** work in a
container.

## What ships and what does not

| Works in a container | Needs extra work |
|---|---|
| `animalpose-cli` — every subcommand | the desktop applications (Qt needs a display) |
| Training, validation, prediction, export | live camera sources (USB/CSI devices are not passed through by default) |
| The 3D stage (calibration, triangulation, rendering) | — |
| Inference on exported artifacts | — |

The GUI is the only real limitation: PySide6 needs an X or Wayland display. Headless servers
should use the CLI, which covers every stage.

## A minimal image

A starting point, not a tested artifact — adjust the base tag to your CUDA version:

```dockerfile
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY . /app

# the extras you actually deploy with; `.[training]` covers train/val/predict/export
RUN pip install --no-cache-dir -e ".[training]"

ENTRYPOINT ["animalpose-cli"]
CMD ["--help"]
```

Build and run:

```bash
docker build -t animalposetracker .
docker run --rm animalposetracker --version
```

The base image already carries PyTorch with CUDA; installing the `training` extra adds
`pycocotools`, and the package itself.

## GPU container

Pass the GPU through and keep the driver on the host:

```bash
docker run --gpus all --rm animalposetracker \
    train --config /work/project/configs/runtime.yaml --device 0
```

`--gpus all` needs the NVIDIA Container Toolkit on the host. Without it, the same command
runs on the CPU and simply takes longer.

## Mounting projects

Keep the project on the host and mount it; nothing in the project layout assumes it lives
inside the image:

```bash
docker run --gpus all --rm \
    -v "$PWD/projects:/work/projects" \
    -v "$PWD/datasets:/work/datasets" \
    animalposetracker \
    train --config /work/projects/demo-me-AnimalRTPose-N-demo/configs/runtime.yaml
```

Paths inside `configs/dataset.yaml` are resolved as written, so either keep them absolute or
mount the dataset at the same prefix it had when the project was created.

## Headless CLI

A full run, no shell:

```bash
docker run --gpus all --rm -v "$PWD:/work" animalposetracker \
    create --dataset trimouse --workspace /work/projects --name demo --worker me \
        --model AnimalRTPose --scale N --date demo

docker run --gpus all --rm -v "$PWD:/work" animalposetracker \
    train --config /work/projects/demo-me-AnimalRTPose-N-demo/configs/runtime.yaml \
        --epochs 50 --batch-size 8 --device 0
```

`training.jsonl` and the checkpoints land on the mounted volume, so the container can be
discarded after each stage.

## Desktop applications

Qt needs a display. On Linux, forward the X socket:

```bash
docker run --rm -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    animalposetracker-gui
```

This needs the `gui` extra and a matching Qt platform plugin in the image. It is rarely worth
the effort — the workspace, annotator and inferencer are easier to run on the host, with the
container reserved for batch jobs.

## Troubleshooting

**`no matching manifest for linux/arm64`.** The upstream PyTorch image may not publish the
tag you picked for ARM. Build on the target architecture, or start from an ARM base and
install PyTorch from the CPU index.

**`CUDA driver version is insufficient`.** The host driver is older than the CUDA runtime in
the image. Use a base tag built for the host's CUDA version, or update the driver.

**Permissions on mounted files.** The container runs as root by default; files it writes
belong to root on the host. Pass your user and group id (`--user $(id -u):$(id -g)`) if that
matters, and make sure the mounted directory is writable by it.
