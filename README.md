<div align="center">

![](docs/assets/logo.png)

![](docs/assets/demos/hand.gif) ![](docs/assets/demos/koi.gif) ![](docs/assets/demos/animals.gif)

[📚 Documentation](docs/getting_started.md) |
<a href="#installation">🛠️ Installation</a> |
<a href="#quick-start">🚀 Quick start</a> |
[🎯 3D](docs/three_d.md) |
[🖥️ GUI](docs/gui.md) |
<a href="https://github.com/wux024/AnimalPoseTracker/issues">🪲 Reporting Issues</a>

<a href="#get-assistance">🫶 Getting Assistance</a> |
<a href="#references--citations">📝 Publications</a> |
[📦 Deployment formats](docs/deployment_formats.md) |
<a href="https://github.com/wux024/AnimalPoseTracker">💻 Source</a>

<a href="https://github.com/wux024/AnimalPoseTracker/actions/workflows/docs.yml"><img src="https://github.com/wux024/AnimalPoseTracker/actions/workflows/docs.yml/badge.svg" alt="Docs build"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-purple" alt="Python 3.10+">
<img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License">
<img src="https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey" alt="Platforms">
<img src="https://img.shields.io/badge/status-0.1.0-orange" alt="Status">
<a href="https://github.com/wux024/AnimalPoseTracker/stargazers"><img src="https://img.shields.io/github/stars/wux024/AnimalPoseTracker.svg?style=social&label=Star" alt="Stars"></a>
<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome">

<img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch 2.0+">
<img src="https://img.shields.io/badge/OpenCV-4.6%2B-5c3ee8" alt="OpenCV 4.6+">
<img src="https://img.shields.io/badge/PySide6-6.2%2B-41cd52" alt="PySide6 6.2+">
<img src="https://img.shields.io/badge/ONNXRuntime-1.10%2B-005ced" alt="ONNXRuntime 1.10+">

</div>

# Welcome! 👋

**AnimalPoseTracker** is a toolbox for markerless pose estimation of animals performing
various behaviors. As long as you can see (label) what you want to track, you can use this
toolbox, as it assumes no particular species, viewpoint, or camera count.
<a href="#why-another-pose-tracker">Read a short development and application summary below</a>.

# Installation

Please follow [Getting started](docs/getting_started.md) — it covers the conda environment,
matching the PyTorch build to your hardware, and installing only the extras for the engine
you deploy with. Python 3.10 or newer is required.

## Quick start

From an empty folder to an exportable model in two steps. The install above is assumed.

- [1] Create a project and train it:

```bash
animalpose-cli create --dataset trimouse --workspace projects --name smoke --worker user \
    --model AnimalRTPose --scale N --date demo --epochs 100 --batch-size 8 --image-size 640
animalpose-cli train --config projects/smoke-user-AnimalRTPose-N-demo/configs/runtime.yaml
```

- [2] Evaluate it, then export the checkpoint:

```bash
animalpose-cli val --config <project>/configs/runtime.yaml \
    --weights <project>/results/train/weights/best.pt
animalpose-cli export --config <project>/configs/runtime.yaml \
    --weights <project>/results/train/weights/best.pt --format onnx --dynamic --opset 17
```

`create` saves only what you pass; everything else keeps the model profile's defaults.
`load` validates a project and prints a summary without opening the GUI or writing files.
Command-line values override `configs/runtime.yaml` for that run and are never written back.

# Documentation: The AnimalPoseTracker Process

![](docs/assets/overview.svg)

`animalpose-cli --help` lists all ten workflows. `animalposetracker`,
`animalpose-annotator` and `animalpose-inferencer` open the three windows; the inferencer also
runs an exported engine (`.onnx`, `.engine`, `.mlpackage`, ...) without a project.

| Step | Command | What you get |
|---|---|---|
| 1. Create | `animalpose-cli create` | Four configs, empty dataset layout |
| 2. Label | `animalpose-annotator` | Boxes, classes, keypoints, skeleton |
| 3. Train | `animalpose-cli train` | `best.pt`, `training.jsonl`, `results.png` |
| 4. Evaluate | `animalpose-cli val` | `metrics.json` — AP, AR, PCK, AUC, EPE |
| 5. Predict | `animalpose-cli predict` | `predictions.json`, rendered video |
| 6. Infer | `animalpose-cli infer` | Live overlay from camera or stream |
| 7. Export | `animalpose-cli export` | `<model>.<format>` + `.json` sidecar |
| 8. 3D | `animalpose-cli calibrate` | Rig intrinsics and extrinsics (ChArUco) |
|  | `animalpose-cli pose3d` | `results/pose3d/...csv` — 3D keypoints |
|  | `animalpose-cli pose3d-render` | 3D video, trajectory plot, summary |

Every command takes `--help`; the flags used most often are in
[Getting started](docs/getting_started.md).

Each step is detailed in [Getting started](docs/getting_started.md); the desktop route
through the same steps is in [Desktop applications](docs/gui.md).

# Usage demo

One run end to end on a built-in dataset — create, train, predict — and the files each
step leaves behind. [Walk through it](docs/demo.md).

# Why another pose tracker?

It continues to be actively maintained, and it aims at a usable `CLI` and `GUI` for
computer-vision researchers and life scientists alike: state-of-the-art models under one
project format, with defaults that work before they are tuned. What sets it apart is the
breadth — one project trains, validates, predicts, exports and reconstructs in 3D, rather
than handing you a model and leaving the rest to glue code.

Reading the two papers below is the shortest way to see what each model is for and how to
adapt it to your setting.

## Performance 🔥

Two model families share one training loop, CLI, validator and export path:

- **AnimalRTPose** — single-stage, real-time; predicts boxes and keypoints in one pass, so it
  runs without a detector. N/S/M/L/X scales.
- **AnimalViTPose** — top-down ViT with a SimCC head; crops each instance from an annotated
  or detected box and maps the prediction back. Small/Base/Large/Huge.

Inference is not limited to these two. One-stage artifacts decode from any of three output
layouts — raw pose head, NMS-applied head, end-to-end head. Two-stage artifacts (SimCC head)
run as well, with `--detector-weights` supplying the boxes they crop from.

**AnimalRTPose** on AnimalPose, 640 x 640 input (Neural Networks, 2025):

<details open><summary>AnimalRTPose accuracy on AnimalPose</summary>

| Model | AP<sub>50:95</sub> | AR<sub>50:95</sub> | Params |
|---|---|---|---|
| AnimalRTPose-N | 72.0 | 80.0 | 2.9 M |
| AnimalRTPose-S | 76.0 | 83.4 | 9.8 M |
| AnimalRTPose-M | 77.2 | 83.9 | 25.3 M |
| AnimalRTPose-L | 77.8 | 84.4 | 47.9 M |
| AnimalRTPose-X | 80.1 | 87.3 | 74.8 M |

</details>

Inference latency on NVIDIA GPUs, in milliseconds:

<details open><summary>Latency on NVIDIA GPUs</summary>

| Model | RTX 2080 Ti | RTX 3090 | A800 |
|---|---|---|---|
| AnimalRTPose-N | 2.1 | 1.3 | 0.9 |
| AnimalRTPose-S | 3.6 | 2.5 | 1.3 |
| AnimalRTPose-M | 7.5 | 4.9 | 2.4 |
| AnimalRTPose-L | 11.3 | 7.5 | 3.7 |
| AnimalRTPose-X | 17.1 | 11.9 | 5.6 |

</details>

The same models on edge devices:

<details open><summary>Latency on edge devices</summary>

| Model | Jetson AGX Orin | Raspberry Pi 5 + AI HAT+ | Atlas 200I |
|---|---|---|---|
| AnimalRTPose-N | 5.1 | 13.0 | 15.6 |
| AnimalRTPose-S | 5.0 | 22.0 | 26.5 |
| AnimalRTPose-M | 6.3 | 33.2 | 53.2 |
| AnimalRTPose-L | 7.5 | 49.3 | 105.7 |
| AnimalRTPose-X | 9.1 | 74.1 | 162.6 |

</details>

**AnimalViTPose** on AP10K, 256 x 256 input (EAAI, 2026):

<details open><summary>AnimalViTPose accuracy on AP10K</summary>

| Model | AP<sub>50:95</sub> | AR | PCK | EPE | Latency |
|---|---|---|---|---|---|
| AnimalViTPose-S | 70.2 | 72.5 | 73.0 | 25.35 | 45.5 ms |
| AnimalViTPose-B | 78.3 | 79.7 | 79.6 | 20.03 | 49.0 ms |
| AnimalViTPose-L | 82.4 | 84.4 | 83.9 | 16.56 | 86.8 ms |
| AnimalViTPose-H | 83.9 | 87.7 | 88.2 | 15.19 | 119.6 ms |

</details>

AnimalPoseTracker inference time on each machine it was tested on:

<details open><summary>AnimalPoseTracker inference time per machine</summary>

| Machine | Compute unit | Inference |
|---|---|---|
| Mac Mini | integrated | 3.2 ms |
| Lenovo laptop | CPU only | 54.5 ms |
| Lenovo desktop | CPU only | 37.7 ms |
| Workstation 1 | RTX 4090 | 6.0 ms |
|  | Intel integrated GPU | 10.0 ms |
|  | Intel NPU | 25.5 ms |
|  | CPU only | 18.2 ms |
| Workstation 2 | RTX 4080 | 2.6 ms |
|  | CPU only | 14.2 ms |
| Jetson AGX Orin | Ampere GPU | 4.2 ms |
| Atlas 200I DK A2 | DAVINCI V300 | 15.6 ms |
</details>

CPUs: Apple M4, Ryzen 7 5800H, Core i5-12400, Intel Ultra 9 285K (both workstations),
ARM Cortex-A78AE and TAISHANV200M on the two edge devices. Workstation 1 is one machine
with four selectable compute units, so its four rows are the same video on the same box.
The edge devices were measured for inference only.

## The history

### Development and applications

The line starts in 2022 with motion parameters read off user-defined keypoints through 3D
pose estimation ([EAAI, 2022](https://doi.org/10.1016/j.engappai.2022.104667)): geometry
first, the network second. Two years later the same idea became a visual perception system
for fish swimming ([IEEE Sensors, 2024](https://doi.org/10.1109/JSEN.2024.3374236)), where the
tracked keypoints describe a behaviour rather than a coordinate.

AnimalRTPose ([Neural Networks, 2025](https://doi.org/10.1016/j.neunet.2025.107685)) moved
the other way — one stage that predicts boxes and keypoints in a single pass — which is what
made cross-species real-time pose affordable on one GPU. The two single-pixel papers of the
same year ([Optics and Lasers, 2025](https://doi.org/10.1016/j.optlaseng.2025.109198);
[Chinese Optics Letters, 2025](https://doi.org/10.3788/COL202523.091101)) moved the input
instead: pose from a sensor that never forms an image. AnimalViTPose
([EAAI, 2026](https://doi.org/10.1016/j.engappai.2025.112749)) then added a feature-map
orthogonal decomposition decoder that lifts accuracy without giving up the deployment path.

None of them assumes a species, a viewpoint or a camera count, and the 3D question has been
there since the first paper. This repository is the system that grew around that line: one
project format, one trainer, one record schema, and the 3D stage that reuses it.

### Ecosystem

Every paper shipped its code where it was trained, and the pieces still line up:

| Paper | Code | What it covers |
|---|---|---|
| EAAI 2022; IEEE Sensors 2024 | [AdamPose](https://github.com/wux024/AdamPose), [AdamPosePlug](https://github.com/wux024/AdamPosePlug) | 3D motion parameters, fish swimming |
| Neural Networks 2025 | [ultralytics fork](https://github.com/wux024/ultralytics/tree/animalrtpose/configs) | AnimalRTPose configs and training |
| Chinese Optics Letters 2025 | [SPCPose](https://github.com/wux024/SPCPose) | pose from single-pixel input |
| EAAI 2026 | [mmpose fork](https://github.com/wux024/mmpose/tree/main/configs/animal_2d_keypoint/animalvitpose) | AnimalViTPose configs |

The forks remain the reference for the original configs; AnimalPoseTracker implements the
same contracts without depending on them.

### Code contributors

AnimalPoseTracker is written and maintained by [Xin Wu](https://github.com/wux024).

# Get assistance

| Where | For | Expectation |
|---|---|---|
| [GitHub Issues](https://github.com/wux024/AnimalPoseTracker/issues) | Bugs, feature requests | Please include the command, the config and the traceback |
| [Discussions](https://github.com/wux024/AnimalPoseTracker/discussions) | Usage questions, setups that work | Community answers |
| wux024@nenu.edu.cn | Collaboration, dataset questions | Best effort |

## References & citations

Please see the [citation page](docs/citation.md) for the two bibtex entries and the
related papers behind the 3D and single-pixel stages.

## License

[AGPL-3.0-or-later](LICENSE). This toolbox builds on several open-source implementations;
the code comments mark the origin of the corresponding modules, and the project is released
under the same license family those modules require.

## Major Versions

For every release, see the [releases page](https://github.com/wux024/AnimalPoseTracker/releases).

**0.1.0** (2026-10) — first public version.

- Two model families sharing one training loop: AnimalRTPose (one-stage, N/S/M/L/X) and
  AnimalViTPose (top-down with a SimCC head, Small/Base/Large/Huge).
- Ten CLI workflows from `create` to `pose3d-render`, plus three desktop applications that
  read and write the same project files.
- An export matrix covering PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, the
  TensorFlow family, PaddlePaddle, MNN, NCNN, IMX500, RKNN and Triton, each behind its own
  extra.
- Multi-camera 3D: ChArUco calibration, index or epipolar association, batched
  triangulation, reprojection and revolving 3D videos.
- Tracking and temporal filtering on pose records, so a recorded sequence can be replayed
  with another tracker without re-running the network.

# News

- **2022** — The line starts: *Motion parameters measurement of user-defined key points using
  3D pose estimation* appears in Engineering Applications of Artificial Intelligence.
- **2024** — *Pose estimation-based visual perception system for analyzing fish swimming* in
  IEEE Sensors Journal, where the tracked keypoints describe a behaviour rather than a
  coordinate.
- **2024-09** — The repository starts as AnimalPoseTracker.
- **2025** — *AnimalRTPose* in Neural Networks; two single-pixel papers follow, moving the
  input from a camera to a sensor that never forms an image.
- **2025-11** — **v0.1.0-alpha** preview: the project workflow, training framework,
  deployment pipeline and desktop GUIs.
- **2026** — *AnimalViTPose* in Engineering Applications of Artificial Intelligence, adding
  the feature-map orthogonal decomposition decoder.
- **2026-10** — **0.1.0**: first public release — two model families, ten CLI workflows,
  three desktop applications, an export matrix, multi-camera 3D with rendering.

# Funding

<!-- TODO: list the grants and institutional support behind this work. -->

