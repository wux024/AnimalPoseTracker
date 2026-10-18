![](https://s3.bmp.ovh/imgs/2025/05/15/e246d2b0dec75c56.png)

# AnimalPoseTracker

**AnimalPoseTracker™️** is a toolbox for cross-species animal pose estimation: a built-in PyTorch training framework, a self-contained deployment pipeline, and desktop GUIs for the whole workflow.

## Features

- **Two model families, one trainer** — AnimalRTPose (real-time, single-stage) and AnimalViTPose (ViT backbone + SimCC, top-down) train through the same loop, CLI and validation.
- **Self-contained training** — project creation, training, validation, dataset prediction, model export and checkpoint I/O run on this project's own workers; no third-party training framework or CLI is required.
- **Wide deployment matrix** — TorchScript, ONNX, OpenVINO, TensorRT, CoreML, the TensorFlow family, PaddlePaddle, MNN, NCNN, Sony IMX500 and Rockchip RKNN, plus a remote NVIDIA Triton model URL.
- **Desktop workflow** — project management, pose annotation, training and live video/camera inference GUIs.
- **Pose analysis tools** — multi-object tracking, temporal filtering and synchronized multi-camera 3D triangulation on top of the pose records every run writes.

## Installation

1. Create a conda environment (Python >= 3.10):
```
conda create -n animalposetracker python=3.10
```

2. [Optional] To train with the built-in PyTorch framework, install a PyTorch 2.0+ build matching your CPU or CUDA setup (https://pytorch.org/get-started/locally/). For example:
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Clone the repository and install with the extras for your environment:
```
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
pip install -v -e ".[training,gui]"    # desktop: GUI + built-in training
pip install -v -e ".[training]"        # headless server: CLI + built-in training
```

Export targets each need their optional runtime or vendor SDK; see the [deployment format support](docs/deployment_formats.md) matrix for requirements and extras such as `.[export]`, `.[export-openvino]` and `.[predict-triton]`.

4. [Optional] Inference engines. AnimalPoseTracker supports six engines: `ONNX`, `OpenVINO`, `TensorRT`, `CoreML`, `CANN`, and `OpenCV`.

- **ONNX** — the widest device range (CPU/GPU/NPU). For NVIDIA GPUs: `pip install onnxruntime-gpu` (after `pip uninstall onnxruntime`); for CPU: `pip install onnxruntime`. The default is the CPU runtime. See https://onnxruntime.ai/docs/install/.
- **OpenVINO** — install the toolkit, e.g. `pip install openvino==2025.1.0`. See https://docs.openvinotoolkit.org/latest/index.html.
- **TensorRT** — install the TensorRT Python API. See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html.
- **CoreML** — `pip install coremltools`. See https://coremltools.readme.io/docs/installation.
- **CANN** — install CANN. See https://www.hiascend.com/document/.
- **OpenCV** — nothing to install; the default engine, CPU only. Build OpenCV from source with the corresponding backend for GPU or other devices.

## Quick start

```bash
conda activate animalposetracker
animalpose-cli create --dataset trimouse --workspace projects --name smoke --worker user \
    --model AnimalRTPose --scale N --date demo --epochs 100 --batch-size 8 --image-size 640
animalpose-cli train  --config projects/smoke-user-AnimalRTPose-N-demo/configs/runtime.yaml
animalpose-cli val    --config <project>/configs/runtime.yaml --weights <project>/runs/train/weights/best.pt
animalpose-cli export --config <project>/configs/runtime.yaml --weights <project>/runs/train/weights/best.pt \
    --format onnx --dynamic --opset 17 --simplify
```

## Command-line workflows

`animalpose-cli` provides unified `train`, `val`, `predict` and `export` commands; run them from the project directory. Each accepts the project configuration plus workflow-specific options:

```bash
animalpose-cli --help
animalpose-cli load projects/smoke-user-AnimalRTPose-N-demo
animalpose-cli predict --config configs/runtime.yaml --weights runs/train/weights/best.pt --split test --conf 0.25 --iou 0.7 --max-detections 300
```

Training options can be overridden for one run without editing `configs/runtime.yaml`:

```bash
animalpose-cli train --config configs/runtime.yaml --epochs 1000 --batch-size 8 --image-size 640 --device 0 --num-workers 8
```

Prediction, tracking, filtering and export options can be passed the same way, e.g.:

```bash
animalpose-cli predict --config configs/runtime.yaml --weights runs/train/weights/best.pt --source images/ --conf 0.4 --no-show-boxes --tracker bytetrack --track-track-buffer 45
animalpose-cli export  --config configs/runtime.yaml --weights runs/train/weights/best.pt --format onnx --image-size 640 --dynamic --opset 17 --no-simplify
animalpose-cli val     --config configs/runtime.yaml --weights runs/train/weights/best.pt --batch-size 8 --image-size 640 --validation-confidence 0.001 --validation-iou 0.7
```

For `train`, `val`, `predict` and `export`, command-line values take precedence over `configs/runtime.yaml` for that run and are not written back to the file. Use `animalpose-cli <command> --help` to see a workflow's options; the commands work without GUI dependencies.

`create` saves only explicitly supplied settings into the new project's `configs/runtime.yaml`; omitted options keep the selected model's defaults. `load` validates a project (directory, `project.yaml`, or `configs/runtime.yaml`) and prints a summary without opening the GUI or modifying files.

`animalpose-cli infer` is the deployment-side runtime: it runs an exported artifact (`.onnx`, `.engine`, `.mlpackage`, ...) on a camera, video file, or stream URL, reading no project configuration. Its own yaml is the whole configuration surface — poser weights, engine and device, source, thresholds, drawing, tracking and filtering, plus the class/keypoint description (`names` / `kpt_shape` / `skeleton`) that decodes the model's output tensor:

```bash
animalpose-cli infer --write-template            # writes ./infer.example.yaml
animalpose-cli infer --config infer.yaml         # the file supplies weights, source, thresholds
animalpose-cli infer --config infer.yaml --conf 0.4   # an explicit flag wins for that run
```

`--write-template -` prints the template instead of writing it. `--poser-weights` and `--source` have no built-in defaults: either the file or the command line must supply them. `infer` refuses `.pt` checkpoints — export first (`animalpose-cli export`), or use `predict` for checkpoints.

`train` exit codes: `0` when training reaches `epochs` **and** when early stopping ends it (an exhausted `early_stopping_patience` is a normal completion — `best.pt` is final), `130` only when the run was interrupted by an external request (`SIGINT`/`SIGTERM` or the `.animalposetracker_training_stop` marker), and `1` on error. Chain `val`/`export` behind `train` in a `set -e` script: early stopping no longer aborts the pipeline. `early_stopping_patience` is counted in **epochs** since the best validation (`epoch - best_epoch`), so it means the same thing at any `validation_interval`; `0` disables early stopping.

AnimalViTPose validation does **not** use flip test-time augmentation by default, so the reported `coco/AP`, `PCK`, `AUC` and `EPE` match what the shipped decoder (`postprocessing/output_decoders.decode_simcc`) produces. Set `validation.flip_tta: true` in the model preset (`animalposetracker/cfg/training.yaml`) to re-enable it, and expect the metrics to read higher than the deployment path.

### Multi-camera 3D

`pose3d` triangulates one `predictions.json` per camera into `results/pose3d/` (`poses3d.json`, optional `poses3d-filter.json`, `report.json`). It needs a `configs/cameras.yaml`; `create --pose3d` writes the template and locks the 3D stage out of public-benchmark projects.

```bash
animalpose-cli calibrate --view camA=frames/camA --view camB=frames/camB --view camC=frames/camC \
    --board-squares 10x7 --square-size 22 --marker-size 16.5 --output configs/cameras.yaml
animalpose-cli pose3d --config configs/runtime.yaml
animalpose-cli pose3d --config configs/runtime.yaml --association epipolar
animalpose-cli pose3d-export --config configs/runtime.yaml --plot results/pose3d/trajectory.png
```

`calibrate` accepts an image folder or a video per view (`--frame-step` samples videos), fits one constrained lens model per camera (or reuses `--intrinsics-from`), and refuses to overwrite an existing calibration without `--force`; `--intrinsic-frames` caps the frames used for the lens fit because OpenCV slows down sharply past ~60. `pose3d-export` writes a flat CSV (`frame, instance, keypoint, x, y, z, confidence, validity, reprojection_error, num_views`), a JSON summary, and an optional trajectory PNG for one instance; `--filtered` reads the filtered sequence.

Instances are associated across cameras with `--association`. `index` (default) pairs the n-th instance of every camera and is only correct with one animal per frame; `epipolar` matches them by the epipolar constraint instead, so detection order, a dropped detection, or two animals crossing cannot swap identities. Each instance in `poses3d.json` records the chosen instance per camera (`view_indices`), and `report.json` carries the match counts under `association_stats`.

The same stage has a home in the GUI: the **3D Reconstruction** tab, which only appears for a project created with the 3D option, lists the cameras (double-click a cell to pick a calibration folder/video or a predictions file) and runs `calibrate`, `pose3d` and `pose3d-export` with their live output in the page. The commands it runs are the CLI's own; the page only collects the paths and numbers. A project created without 3D can gain it later from **Tools > Enable 3D reconstruction for this project**, which writes the flag, the empty `configs/cameras.yaml` and `results/pose3d` without touching anything else; public-benchmark projects stay locked out.

## Configuration model

Training defaults live in [`animalposetracker/cfg/training.yaml`](animalposetracker/cfg/training.yaml). Values identical across AnimalRTPose and AnimalViTPose are defined once under `training.shared`; each model profile contains only its specific or differing training, loss, augmentation and validation values. The model type in `project.yaml` selects the matching key under `training.models`.

- **`configs/model.yaml`** — the network graph: AnimalViTPose scale variants, SimCC head geometry, input normalization.
- **`configs/dataset.yaml`** — dataset paths and keypoint metadata. The dataset's shape (class names, keypoint names, counts, skeleton, flip table, OKS sigmas) lives here as its single home.
- **`project.yaml`** — identity only: project name, worker, model type/scale, date, sources. `model.yaml` repeats `kpt_shape` as the model output contract, which must match the dataset.
- **`configs/runtime.yaml`** — every project's runtime parameters, sectioned by flow (`common` / `train` / `predict` / `export`) as a snapshot of the registered keys; the full key reference is generated into [`docs/configuration.md`](docs/configuration.md).

AnimalViTPose uses the same built-in trainer as AnimalRTPose, with a top-down instance-crop data adapter and the SimCC/KL loss, optimizer, schedule, and COCO/PCK/AUC/EPE evaluation settings from the [AnimalViTPose recipe](https://github.com/wux024/mmpose/blob/main/configs/animal_2d_keypoint/animalvitpose/ap10k/animalvitpose-small_8xb64-210e_ap10k-256x256.py). The Small/Base/Large/Huge scales and project image size select the ViT graph inside the same trainer. The AnimalViTPose profile defaults to the matching MAE ViT checkpoint and caches it under the user's local cache directory; set `pretrained: false` to train from scratch, or provide `pretrained_weights` for a local or remote checkpoint.

Both models read normalized YOLO-Pose TXT labels or COCO Keypoints JSON. AnimalViTPose can validate YOLO labels by building COCO ground truth in memory. For COCO, point the split image fields at the image folders and provide the split annotation files:

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

With this layout, the trainer automatically reads `annotations/train.json`, `val.json`, and `test.json`; standard `person_keypoints_{split}2017.json` names are also detected. Explicit `train_annotations` and `val_annotations` paths can override discovery. COCO category names map to the configured class names. Images, boxes, and keypoints are read directly from JSON; training does not convert the dataset or import any third-party training framework.

For keypoint AP, configure one positive OKS sigma per landmark with `kpt_oks_sigmas` in the dataset YAML. This replaces COCOeval's built-in human COCO-17 vector. When bringing in an MMPose dataset metainfo, write its `sigmas` values under this field. If omitted, the trainer warns and uses the uniform `1 / number_of_keypoints` custom-keypoint fallback; animal datasets should provide their own vector when landmark tolerances differ.

Project evaluation uses the native COCO/PCK/AUC/EPE validator. Project prediction defaults to the configured `test` split, then `val`; AnimalRTPose runs on full images, while AnimalViTPose crops each annotated instance from the selected COCO/YOLO split and maps SimCC predictions back to the source image. The predictor exposes an `InstanceBoxProvider` interface for a future detector; live camera/video inference remains the separate inferencer workflow. Video project predictions can optionally chain tracking and temporal filtering, and synchronized camera outputs can be triangulated through the project API; see [the pose analysis workflow guide](docs/pose_analysis_workflows.md). Native TorchScript export supports both model heads; ONNX export and simplification use the optional `.[export]` dependencies.

## GUI workflows

### Train and test a model

1. Activate the environment and start the toolbox: `animalposetracker`

2. Create a new project (custom dataset or public datasets), or load an existing one.

https://github.com/user-attachments/assets/156b6e16-5660-4c81-b14b-73120b67417a

https://github.com/user-attachments/assets/7c850634-9aef-4b72-95f3-1b11c491a25c

3. Manage the project configuration, then extract frames from videos (automatic or manual) and label poses with 'animalpose-annotator'.

https://github.com/user-attachments/assets/c48b6511-7979-4a84-ac59-b9ba4d6edebe

https://github.com/user-attachments/assets/61662bd7-4ee4-4266-92f1-6f22bedcac3b

https://github.com/user-attachments/assets/6ec6ac35-b6c9-41ee-8107-403861ec054a

4. Train a model on the labeled data with the training configuration.

https://github.com/user-attachments/assets/79636a6e-3813-40f1-8595-06040e2fe178

5. Evaluate the trained model on the test set, then predict on videos or the `test` set. For deployed models, use the `animalpose-inferencer` plugin.

https://github.com/user-attachments/assets/f352d5fe-1303-4070-b810-b26769faab28

https://github.com/user-attachments/assets/ba5d9c05-73ec-44ae-b4b1-8100f02f1998

6. Export the trained model to different inference engines.

https://github.com/user-attachments/assets/4abf70c0-e277-4860-ae12-a87e3051a3c8

### Inference (videos and cameras)

1. Start the inferencer: `animalpose-inferencer`

2. Choose the configuration file and the poser weights file; available inference engines and devices appear automatically.

https://github.com/user-attachments/assets/db726dd8-a092-40e3-94ea-e3d423973a03

3. Enable camera or video input, pick a local video or a camera, and preview it.

https://github.com/user-attachments/assets/007caeb6-b5b2-48a8-9873-58ff9db002b7

4. Set camera parameters (resolution, FPS, model bits, engine, device) — the camera must support the chosen resolution and FPS.

https://github.com/user-attachments/assets/e67da907-e765-4247-affb-b85563a33d28

5. Start inference and tune the run controls: confidence, visualization, tracking and filtering.

https://github.com/user-attachments/assets/0b57bba6-9ed6-47b1-9f4f-1d449762c784

https://github.com/user-attachments/assets/9d2ad685-7201-4488-a653-500a6c72b3bf

The configuration file describes the classes and keypoints. Either style is accepted: a project yaml (`classes_name` / `keypoints_name` / `skeleton`) or a dataset yaml (`names` / `kpt_shape` / `skeleton`). These keys are not only labels — their counts tell the decoder how the model's output tensor is laid out, so a configuration that disagrees with the artifact shifts the keypoint block and decodes nonsense. Both files are checked as soon as they are picked:

- missing class names or keypoint count: a message explains which key is missing, and Start stays disabled;
- keypoint count disagrees with the artifact metadata: a warning is printed and the run is still allowed, because the decoder follows the configuration;
- AnimalViTPose SimCC artifacts are refused while selecting — this toolbox infers YOLO-Pose layouts only; run those through `animalposetracker predict`.

Tracking, filtering and records:

- **Track** — tick `Track` and pick a tracker; every label then carries its track id (`#1 mouse: 0.90`). The list is the same one `infer --tracker` accepts, and a tracker follows the pose records rather than the pixels, which is why the same run can be replayed later with another algorithm.
- **Records** — the per-frame pose records are saved with every run to `runs/inference/records.json`, in the same `[{frame, predictions}, ...]` shape `infer --records` writes, with a `track_id` on every detection when tracking was on. There is no switch for it: the records are the run's result. `Save Video` writes the rendered frames to `runs/inference/output.avi`.
- **Filter** — tick `Filter` and choose a method (`median` or `savgol`) with its window, order, confidence and gap settings. The smoothing runs over the whole sequence when the run ends; the records file holds the smoothed records. Filtering needs track ids, so the filter controls stay disabled until `Track` is ticked, exactly as `--pose-filter` requires `--tracker` on the command line.

### Annotate poses

1. Start the annotator: `animalpose-annotator`

2. Choose the configuration file and the images directory.

https://github.com/user-attachments/assets/7d820fbe-390e-4387-9678-ac229e930cda

3. Draw bounding boxes, choose the object class, draw keypoints and skeleton, then save or export the annotations.

https://github.com/user-attachments/assets/a22b28df-4b12-48b0-a087-4ab07f6ea17f

## Documentation

| Document | Contents |
|---|---|
| [configuration.md](docs/configuration.md) | Full reference of every registered configuration key (generated from the schema) |
| [deployment_formats.md](docs/deployment_formats.md) | Export/prediction format matrix and per-format installation extras |
| [pose_analysis_workflows.md](docs/pose_analysis_workflows.md) | Video tracking/filtering API and synchronized multi-camera 3D triangulation |
| [project_config_example/](docs/project_config_example/) | A complete example project configuration |

## Citation

```
@article{Wu2026, 
   title = {Cross-species animal pose estimation via feature map orthogonal decomposition decoder}, 
   journal = {Engineering Applications of Artificial Intelligence}, 
   volume = {163}, 
   pages = {112749}, 
   year = {2026}, 
   doi = {https://doi.org/10.1016/j.engappai.2025.112749}, 
   author = {Xin Wu and Yanmei Wang and Lianming Wang and Jipeng Huang}, 
   keywords = {Animal pose estimation, Vision transformer, Feature map decomposition}
   }
@article{Wu2025,
   title = {AnimalRTPose: Faster cross-species real-time animal pose estimation},
   journal = {Neural Networks},
   volume = {190},
   pages = {107685},
   year = {2025},
   issn = {0893-6080},
   doi = {https://doi.org/10.1016/j.neunet.2025.107685},
   author = {Xin Wu and Lianming Wang and Jipeng Huang},
   keywords = {Animal pose estimation, Real-time, Cross-species, Lightweight network},
}
```

## License

[AGPL-3.0-or-later](LICENSE). This toolbox builds on several open-source implementations; the code comments mark the origin of the corresponding modules, and the project is released under the same license family those modules require.
