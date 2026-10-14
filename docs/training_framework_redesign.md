# Self-contained training framework redesign (draft)

- owner: Codex
- date: 2026-10-08
- status: in progress

## Confirmed requirements

- The author's explicit requirement: model training must not depend on the YOLO training framework.
- The alignment baseline is fixed at [`wux024/ultralytics@188d5bdb`](https://github.com/wux024/ultralytics/tree/188d5bdb204263ced5f061aeabd4d1606a150911): behaviour follows that commit's pose validator, metrics and loss. This project implements the corresponding logic locally and does not import Ultralytics in the training runtime.

## Current implementation

- New self-contained training configuration, PyTorch training loop, parameter grouping and `optimizer=auto` selection, learning-rate schedule, warmup, `nbs` gradient accumulation, AMP, early stopping and EMA checkpoint recovery.
- The local `YOLOPoseHead` loss is aligned with the `animalrtpose` branch semantics: task-aligned assignment, CIoU, DFL, OKS keypoint loss, visibility BCE and configured weights. That branch is not imported at runtime.
- The existing split directories and keypoint text labels are read by the local dataset loader. The training side wires in 2x2 Mosaic, HSV, rotation/translation/scale/shear/perspective, MixUp and flips; crop transforms update boxes, keypoints and visibility together. When a dataset provides no `flip_idx`, horizontal/vertical flips are disabled with a warning.
- The `Tri-Mouse` preset carries `flip_idx=[0,2,1,3,...,11]`; the mouse smoke config now includes that mapping, and the training data loader reads it and reorders left/right keypoints.
- With `annotation_format: coco`, the loader discovers `annotations/{train,val,test}.json` or `person_keypoints_{split}2017.json` per split; `train_annotations`/`val_annotations` can override the paths explicitly.
- `cache` supports RAM and disk caching, `fraction` draws from the training set only, and `close_mosaic` disables Mosaic by training epoch. `time` caps training in hours; `profile` records data wait, forward/loss, backward/optimizer and epoch time.
- `multi_scale` randomly scales training batches by model stride; `freeze` freezes the first N layers or explicit layer indices while keeping frozen-layer BN statistics unchanged; `single_cls` maps both data labels and the model output head to a single class. A non-zero `copy_paste` on pose tasks raises an explicit error, because the reference parameter only defines segmentation copy-paste.
- AnimalRTPose and AnimalViTPose both compute keypoint AP through COCOeval's `keypoints` evaluator; box AP no longer enters the pose validation score. Both report `coco/AP`, `coco/AP50`, `coco/AP75` and `coco/AR`, and `coco/AP` selects the best checkpoint. AnimalRTPose keeps `fitness` and the old pose mAP fields as compatibility aliases.
- `animalposetracker/cfg/training.yaml` defines training-side defaults only. Parameters identical across both models live once under `training.shared`; values that differ or belong to one model stay in its profile. The training profile is selected from the single `model_type` in `project.yaml`, and the model choice is not duplicated in `other.yaml`; the model graph, the AnimalViTPose scale, SimCC output geometry and input normalization belong to `configs/model.yaml`; data paths and keypoint metadata belong to `configs/dataset.yaml`. The project layer flattens the selected training profile in memory only; GUI source code needs no changes.
- `project.yaml` keeps the class/keypoint names, counts and skeleton metadata the annotation GUI reads directly; `dataset.yaml` provides the corresponding fields the data loader needs; `model.yaml.kpt_shape` is the model output shape contract and must match the dataset. `kpt_oks_sigmas` lives only in `dataset.yaml`, and the model choice only in `project.yaml`.
- Project evaluation runs through `animalposetracker.training.cli --validate-only`, reusing the training validator and writing COCO/PCK/AUC/EPE to `runs/val/metrics.json`. Project prediction defaults to the test split (val when missing): AnimalRTPose predicts on full images, AnimalViTPose crops each annotated instance from YOLO/COCO boxes and maps predictions back to the source image, with the `InstanceBoxProvider` interface reserved for a future detector. The prediction worker does not call the GUI's live inference engine.
- Project export runs through `animalposetracker.export.cli` without the YOLO CLI. TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel/GraphDef/TFLite/Edge TPU/TF.js, PaddlePaddle, MNN, NCNN, IMX500 and RKNN have local conversion dispatch; target SDKs/converters install optionally per format. Exported models carry JSON metadata describing input size, normalization, keypoint shape and output tensors.
- Dataset YAMLs record per-keypoint OKS sigmas via `kpt_oks_sigmas` only, passed explicitly to COCOeval to override its built-in human COCO-17 vector. MMPose dataset metainfo `sigmas` are written into `kpt_oks_sigmas` when migrating into this project. The uniform `1/K` custom-keypoint fallback applies when absent; custom animal data should configure its own sigmas. `plots` writes training and validation curves.
- The GUI's train/resume entry points call `python -m animalposetracker.training.cli`; training status is written to JSONL and shown on the status bar, and stop requests travel through a marker file so checkpoints are saved.
- The GUI's `box/cls/dfl/pose/kobj`, `val` and `patience` now map to the self-contained loss weights, validation switch and early stopping; the static form defaults are aligned with `cfg/default.yaml` too.
- Training and export dependency extras and subpackage discovery were added so the new training packages install with the project. The README does not require installing Ultralytics.
- DDP uses PyTorch `DistributedDataParallel` and the distributed training sampler; two-process CPU/Gloo training has run on a mouse subset. Linux prefers NCCL, Windows uses Gloo; the current work environment is CPU-only and the GPU path is untested. Distributed validation still computes the full validation set on every rank -- correct but redundant.
- AnimalViTPose joins the same `Trainer`: top-down crops from COCO or YOLO instance boxes, with MMPose Small/Base/Large/Huge. Network width, depth, drop-path, the MAE pretrained URL, SimCC output geometry and normalization parameters live in the model YAML; layer decay and the training recipe, loss, augmentation and validation parameters live in the training configuration. Both COCO and YOLO validation use COCOeval; YOLO labels are converted to COCO ground truth in memory. Validation reports AP, PCK, AUC and EPE. AnimalRTPose also reports PCK, AUC and EPE, with errors counted only over instances COCOeval matched at OKS 0.50; unmatched instances show through COCO AP/AR. MMPose serves as an implementation baseline only and is not imported in the training runtime.
- The `SimCCHead` top-down data adapter, loss and metrics join the shared training loop through pluggable components; no separate optimizer-step / checkpoint / resume loop was copied. `profile` remains local training-stage timing and does not run ONNX/TensorRT benchmarks.
- Two CPU epochs were run after alignment on the desktop `mouse` data and `animalrtpose-n.pt`. The loader maps known old checkpoint class names to local network classes with PyTorch's restricted loading (no YOLO package import); 423 of the model's 465 tensors transferred, and the 42 differently-shaped classification/keypoint-head tensors keep their new initialization.

## Design boundaries

1. Training, resume, validation, project-dataset prediction, model export and checkpoint I/O are all orchestrated by AnimalPoseTracker itself; these entry points do not import or call Ultralytics, do not call the `yolo` command, and do not reuse the GUI's live inference engine.
2. Networks are built from the project's own `animalposetracker.nn`, with gradient updates, optimizers, learning-rate schedules, mixed precision and device management implemented in PyTorch.
3. Model definitions, dataset reading, model-specific target encoding and losses, the shared training loop, metrics, checkpoints and GUI process management are split into explicit modules.
4. Project YAMLs, dataset split directories, label texts and `runs/train/weights/{best,last}.pt` are compatibility interfaces kept first. Reusing a label file format means only parsing that format -- not depending on the framework that produced or trained on those labels.
5. When an old project first opens, its original configuration is read; new training configuration needs explicit defaults and migration logic so author project parameters are never silently overwritten. Which old weights can initialize and whether old optimizer states can resume need to be handled per checkpoint kind.

## Proposed code boundaries

- `animalposetracker/training/config.py`: training parameter model, type coercion, range validation, and mapping from old project parameters to the new configuration.
- `animalposetracker/training/data/`: read existing images and labels, validate samples, build batches, transform images. The unified batch carries images, target boxes, classes, keypoints and visibility.
- `animalposetracker/training/losses/`: register independent criteria per network head. The multi-scale regression head implements its own candidate assignment, box distribution regression and keypoint targets; the SimCC head implements keypoint cropping and x/y distribution targets. The shared training loop does not inspect concrete tensor shapes.
- `animalposetracker/training/engine.py`: training and validation loops, optimizer, learning-rate schedule, gradient clipping, mixed precision, early stopping and resume.
- `animalposetracker/training/checkpoint.py`: save model weights, optimizer and schedule state, epoch, RNG state and configuration; also provide a weights-only loading entry for inference.
- `animalposetracker/training/metrics.py` and `callbacks.py`: central definitions of keypoint metrics, logging, checkpoint and progress events.
- `animalposetracker/training/cli.py`: AnimalPoseTracker's own command-line training worker. The GUI starts this worker and receives parseable status events, without depending on third-party training commands.

## Proposed implementation order

1. First fix the interfaces from project configuration to training parameters, labels to batches, and network heads to losses, and settle `pyproject.toml` subpackage discovery and the optional installation of training dependencies.
2. Implement data reading for the existing label formats and single/multi-instance batches; do not change the GUI or annotator data formats first.
3. Complete the shared training loop and checkpoints, then wire in losses per model head. Both model-specific paths (`AnimalRTPose` and `AnimalViTPose`) are already connected to the shared loop; the remaining existing model configs follow in actual use order.
4. Add self-contained validation, logging, pause/stop signals and resume entry points, keeping the current project output-directory conventions. Self-contained validation and resume are connected.
5. Switch project validation, AnimalRTPose/AnimalViTPose dataset prediction and model export to the local worker. Done; AnimalViTPose prediction on external images/videos can inject an `InstanceBoxProvider` into the worker.

## Points that need special care during implementation

- `AnimalViTPose` regresses top-view pose from a single-animal crop, unlike the multi-object detection-style data path. Its cropping, keypoint coordinate transforms and training/inference preprocessing must use the same geometry rules.
- `AnimalViTPose` is now in `MODEL_YAML_PATHS`; a project uses one fixed `configs/model.yaml`, and the project scale lives in `project.yaml` and is applied to that model graph at training time.
- The current YAMLs are written with a graph structure similar to common detection networks; the runtime parser and graph builder are local implementations. The redesign should state that this is the project's own model description format and avoid implicit dependencies on third-party parsers or training runtimes.
- Training, validation, prediction and export reuse `other.yaml` configuration but execute through separate local Python workers. AnimalViTPose is a top-down model; dataset prediction defaults to the split's annotation boxes, while external images/videos need an `InstanceBoxProvider` injected into the worker. The full image must never be silently treated as one instance crop.
