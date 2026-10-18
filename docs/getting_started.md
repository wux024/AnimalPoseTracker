# Getting started

Everything needed to install AnimalPoseTracker, create a project, train a model and
run it on new data. The 3D stage has its own page: [Multi-camera 3D](three_d.md).

## Requirements

- Python 3.10 or newer.
- A PyTorch 2.0+ build matching your CPU or CUDA setup, if you want to train with the
  built-in framework (<https://pytorch.org/get-started/locally/>). Inference on exported
  artifacts needs no PyTorch at all.
- Optional per-format runtimes only when you export to them (see
  [Deployment formats](deployment_formats.md)).

## Installation

Create an environment and install the package in editable mode with the extras you need:

```bash
conda create -n animalposetracker python=3.10
conda activate animalposetracker

# PyTorch, matching your hardware; this line is CUDA 11.8 as an example
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
pip install -v -e ".[training,gui]"    # desktop: GUI + built-in training
pip install -v -e ".[training]"        # headless server: CLI + built-in training
```

Extras are independent: `.[export]` adds the ONNX tooling, `.[export-openvino]` the
OpenVINO tooling, `.[predict-triton]` the Triton client, and so on. The
[format matrix](deployment_formats.md) lists every target with the extra it needs.

### Inference engines

Inference supports six engines. Only the one you deploy with has to be installed.

| Engine | Install | Notes |
|---|---|---|
| ONNX | `pip install onnxruntime` (CPU) or `onnxruntime-gpu` after `pip uninstall onnxruntime` | Widest device range: CPU, GPU, NPU |
| OpenVINO | `pip install openvino==2025.1.0` | Intel CPU/GPU/NPU |
| TensorRT | TensorRT Python API | NVIDIA GPUs, fastest path |
| CoreML | `pip install coremltools` | Apple silicon and macOS |
| CANN | Ascend CANN toolkit | Huawei Ascend |
| OpenCV | nothing to install | CPU only; build OpenCV with the backend for GPU |

## Quick start

```bash
animalpose-cli create --dataset trimouse --workspace projects --name smoke --worker user \
    --model AnimalRTPose --scale N --date demo --epochs 100 --batch-size 8 --image-size 640
animalpose-cli train  --config projects/smoke-user-AnimalRTPose-N-demo/configs/runtime.yaml
animalpose-cli val    --config <project>/configs/runtime.yaml \
    --weights <project>/results/train/weights/best.pt
animalpose-cli export --config <project>/configs/runtime.yaml \
    --weights <project>/results/train/weights/best.pt \
    --format onnx --dynamic --opset 17 --simplify
```

`create` saves only the settings you pass into the new project's `configs/runtime.yaml`;
everything else keeps the selected model profile's defaults. `load` validates a project
(directory, `project.yaml`, or `configs/runtime.yaml`) and prints a summary without
opening the GUI or writing anything.

## Command-line workflows

`animalpose-cli` is the whole workflow surface: `create`, `load`, `train`, `val`,
`predict`, `infer`, `export`, `pose3d`, `pose3d-render`, `calibrate`. Run them from the
project directory.

```bash
animalpose-cli --help
animalpose-cli load projects/smoke-user-AnimalRTPose-N-demo
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --split test \
    --conf 0.25 --iou 0.7 --max-detections 300
```

Command-line values take precedence over `configs/runtime.yaml` for that run and are not
written back, so one-off overrides never disturb the project:

```bash
animalpose-cli train --config configs/runtime.yaml \
    --epochs 1000 --batch-size 8 --image-size 640 --device 0 --num-workers 8
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source images/ --conf 0.4 \
    --no-show-boxes --tracker bytetrack --track-track-buffer 45
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx --dynamic --opset 17
```

`animalpose-cli <command> --help` lists that workflow's options; none of these commands
need the GUI extras.

### Training exit codes

`train` exits `0` both when it reaches `epochs` and when early stopping ends it — an
exhausted `early_stopping_patience` is a normal completion, and `best.pt` is final. It
exits `130` only when the run was interrupted from outside (`SIGINT`/`SIGTERM`, or the
`.animalposetracker_training_stop` marker), and `1` on error. Chain `val`/`export` behind
`train` in a `set -e` script: early stopping no longer aborts the pipeline.

`early_stopping_patience` counts **epochs since the best validation** (`epoch - best_epoch`),
so it means the same thing at any `validation_interval`; `0` disables early stopping.

### Validation and test-time augmentation

AnimalViTPose validation does **not** use flip test-time augmentation by default, so the
reported `coco/AP`, `PCK`, `AUC` and `EPE` match what the shipped decoder
(`postprocessing/output_decoders.decode_simcc`) actually produces. Set
`validation.flip_tta: true` in the model profile to re-enable it, and expect the metrics to
read higher than the deployment path.

## Deployment-side inference

`animalpose-cli infer` runs an exported artifact (`.onnx`, `.engine`, `.mlpackage`, ...) on
a camera, a video file or a stream URL. It reads no project configuration: its own yaml is
the whole surface — poser weights, engine and device, source, thresholds, drawing, tracking
and filtering, plus the class/keypoint description (`names` / `kpt_shape` / `skeleton`)
that decodes the output tensor.

```bash
animalpose-cli infer --write-template              # writes ./infer.example.yaml
animalpose-cli infer --config infer.yaml           # the file supplies weights, source, thresholds
animalpose-cli infer --config infer.yaml --conf 0.4   # an explicit flag wins for that run
```

`--write-template -` prints the template instead of writing it. `--poser-weights` and
`--source` have no defaults: the file or the command line must supply them. `infer`
refuses `.pt` checkpoints — export first, or use `predict` for checkpoints.

## Configuration

Project defaults live in [`animalposetracker/cfg/training.yaml`](../animalposetracker/cfg/training.yaml).
Values shared by both model families are defined once under `training.shared`; each model
profile carries only what differs. `project.yaml`'s model type selects the matching key
under `training.models`.

| File | Holds |
|---|---|
| `project.yaml` | Identity only: project name, worker, model type/scale, date, sources, `pose3d` / `public_dataset` flags |
| `configs/dataset.yaml` | Dataset paths and the dataset's shape — class names, keypoint names, counts, skeleton, flip table, OKS sigmas |
| `configs/model.yaml` | The network graph: scale variants, SimCC head geometry, input normalization |
| `configs/runtime.yaml` | Every runtime parameter, sectioned by flow (`common` / `train` / `predict` / `export`) |
| `configs/cameras.yaml` | Multi-camera calibration, only in projects created with the 3D option |

The full key reference is generated from the registry: [configuration.md](configuration.md).

## Project layout

A project directory is `name-worker-model-scale-date`. Only the four configs and a trained
checkpoint are required; every other directory appears when its stage runs.

```text
<project>/
├── project.yaml                  identity: name, worker, model, date, sources, flags
├── configs/
│   ├── dataset.yaml              shape: keypoints, classes, skeleton, OKS sigmas
│   ├── model.yaml                nc / kpt_shape / backbone / head
│   ├── runtime.yaml              every runtime parameter, by flow
│   └── cameras.yaml              multi-camera calibration (3D projects only)
├── datasets/                     training data; may point outside the project
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/  YOLO txt
│   └── annotations/              COCO json when that format is used
├── sources/                      raw material: images, videos, extracted frames
├── pretrained/                   cached pretrained checkpoints
└── results/
    ├── train/                    weights/{best,last}.pt, training.jsonl, metrics.json, results.png
    ├── val/                      training.jsonl, metrics.json
    ├── predict/<camera>/         predictions.json, tracked_predictions.json,
    │                             predictions-filter.json, prediction_metadata.json, images/, videos/
    ├── pose3d/                   poses3d.json, poses3d-filter.json, report.json,
    │                             poses3d.csv, videos/, images/
    └── export/                   <model>.<format> plus its .json sidecar (preprocessing contract)
```

`training.jsonl` is the machine-readable event stream; `metrics.json` holds the final
metrics; `prediction_metadata.json` records the model file, its format, the split and the
tracker a prediction run used. The `.json` beside an exported artifact is its sidecar: it
carries the preprocessing, NMS and class contract, and the inferencer refuses an artifact
whose sidecar disagrees with the configuration.

## Datasets

Both models read normalized YOLO-Pose TXT labels or COCO Keypoints JSON. AnimalViTPose can
validate YOLO labels by building COCO ground truth in memory.

For COCO, point the split image fields at the image folders and let the annotations be
discovered:

```yaml
annotation_format: coco
path: /datasets/mouse
train: images/train
val: images/val
test: images/test
kpt_shape: [12, 3]
names:
  0: mus1
  1: mus2
  2: mus3
skeleton: [[0, 1], [0, 2], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
```

The trainer then reads `annotations/train.json`, `val.json` and `test.json`
(`person_keypoints_{split}2017.json` names are detected too); `train_annotations` and
`val_annotations` override discovery. Images, boxes and keypoints are read straight from
JSON — training never converts the dataset or imports another framework.

### OKS sigmas

Keypoint AP needs one positive OKS sigma per landmark, set as `kpt_oks_sigmas` in the
dataset YAML. This replaces COCOeval's built-in human COCO-17 vector; when importing an
MMPose dataset metainfo, copy its `sigmas` here. If omitted, the trainer warns and falls
back to the uniform `1 / number_of_keypoints` — animal datasets with different landmark
tolerances should provide their own.

## Evaluation and prediction

Project evaluation runs the native COCO/PCK/AUC/EPE validator. Project prediction defaults
to the configured `test` split, then `val`: AnimalRTPose runs on full images, while
AnimalViTPose crops each annotated instance from the selected split and maps SimCC
predictions back to the source image. The predictor exposes an `InstanceBoxProvider`
interface for a future detector; live camera/video inference stays the separate inferencer
workflow.

Video predictions can chain tracking and temporal filtering, and synchronized camera
outputs can be triangulated through the project API — see
[Pose analysis workflows](pose_analysis_workflows.md).
