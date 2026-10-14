![](https://s3.bmp.ovh/imgs/2025/05/15/e246d2b0dec75c56.png)

# Welcome! 👋

**AnimalPoseTracker™️** is a toolbox for cross-species animal pose estimation. 

# Installation: How to install AnimalPoseTracker

1. Create a conda environment:
We recommend creating a new conda environment for AnimalPoseTracker, recommended Python >= 3.10. You can do this by running the following command in your terminal:
```
conda create -n animalposetracker python=3.10
```

2. [Optional] To train with AnimalPoseTracker's built-in PyTorch framework, install a PyTorch 2.0+ build that matches your CPU or CUDA setup. Follow the instructions on the official website: https://pytorch.org/get-started/locally/. For example, if you are using conda, you can run the following command:
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Install AnimalPoseTracker with the optional dependencies for your environment.
Clone the repository:
```
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
```
For a desktop environment with the GUI and built-in training:
```
pip install -v -e ".[training,gui]"
```
For a headless server that uses the CLI and built-in training, omit the GUI dependencies:
```
pip install -v -e ".[training]"
```

Project training, validation, prediction, and model export use AnimalPoseTracker's own workflows; they do not require the YOLO training framework or CLI. Install `.[training]` for PyTorch and COCO evaluation support. Export targets include TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel/GraphDef/Lite/Edge TPU/JS, PaddlePaddle, MNN, NCNN, IMX500, and RKNN. Project prediction also accepts a remote Triton model URL; TensorFlow.js is export-only, matching the Ultralytics AutoBackend baseline. Each target needs its optional runtime or vendor SDK; see [deployment format support](docs/deployment_formats.md) for the matrix and installation extras.

AnimalViTPose uses the same built-in trainer as AnimalRTPose, with a top-down instance-crop data adapter and the SimCC/KL loss, optimizer, schedule, and COCO/PCK/AUC/EPE evaluation settings from the [MMPose AnimalViTPose recipe](https://github.com/wux024/mmpose/blob/main/configs/animal_2d_keypoint/animalvitpose/ap10k/animalvitpose-small_8xb64-210e_ap10k-256x256.py). The MMPose Small/Base/Large/Huge scales and project image size select the ViT graph inside the same trainer. MMPose itself is not imported at training time; install `.[training]` for PyTorch and COCO evaluation support.

Built-in training defaults are defined in [`animalposetracker/cfg/training.yaml`](animalposetracker/cfg/training.yaml). Values identical across AnimalRTPose and AnimalViTPose are defined once under `training.shared`; each model profile contains only its specific or differing training, loss, augmentation and validation values. The model type in `project.yaml` selects the matching key under `training.models`, so the selection is stored only once. Network graphs, AnimalViTPose scale variants, SimCC head geometry and input normalization stay in the single selected project's `configs/model.yaml`; dataset paths and keypoint metadata stay in `configs/dataset.yaml`. The selected scale comes from `project.yaml` and is applied to the model graph at training time. Every project's runtime parameters live in `configs/runtime.yaml`, sectioned by flow (`common` / `train` / `predict` / `export`) as a snapshot of the registered keys; the full key reference is generated into [`docs/configuration.md`](docs/configuration.md).

The dataset's shape -- class names, keypoint names, counts, skeleton, flip table, OKS sigmas -- lives in `dataset.yaml`, its single home, and the annotation pages read it there. `model.yaml` repeats `kpt_shape` as the model output contract, which must match the dataset. `project.yaml` keeps identity only (project name, worker, model type/scale, date, sources).

The AnimalViTPose profile defaults to the matching MAE ViT checkpoint and caches it under the user's local cache directory. Set `pretrained: false` to train from scratch, or provide `pretrained_weights` to use a local or remote checkpoint. Project evaluation uses the native COCO/PCK/AUC/EPE validator. Project prediction defaults to the configured `test` split, then `val`; AnimalRTPose runs on full images, while AnimalViTPose crops each annotated instance from the selected COCO/YOLO split and maps SimCC predictions back to the source image. The predictor exposes an `InstanceBoxProvider` interface for a future detector; live camera/video inference remains the separate inferencer workflow. Video project predictions can optionally chain tracking and temporal filtering, and synchronized camera outputs can be triangulated through the project API; see [the pose analysis workflow guide](docs/pose_analysis_workflows.md). Native TorchScript export supports both model heads; ONNX export and simplification use the optional `.[export]` dependencies.

Both AnimalRTPose and AnimalViTPose read normalized YOLO-Pose TXT labels or COCO Keypoints JSON. AnimalViTPose can validate YOLO labels by building COCO ground truth from them in memory. For COCO, point the split image fields at the image folders and provide the split annotation files:
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
With this layout, the trainer automatically reads `annotations/train.json`, `val.json`, and `test.json`; standard `person_keypoints_{split}2017.json` names are also detected. Explicit `train_annotations` and `val_annotations` paths can override discovery. COCO category names map to the configured class names. Images, boxes, and keypoints are read directly from JSON; training does not convert the dataset or import Ultralytics.

For keypoint AP, configure one positive OKS sigma per landmark with `kpt_oks_sigmas` in the dataset YAML. This replaces COCOeval's built-in human COCO-17 vector. When bringing in an MMPose dataset metainfo, write its `sigmas` values under this field. If omitted, the trainer warns and uses the uniform `1 / number_of_keypoints` custom-keypoint fallback; animal datasets should provide their own vector when landmark tolerances differ.

4. [Optional] AnimalPoseTracker supports six inference engines: `ONNX`, `OpenVINO`, `TensorRT`, `CoreML`, `CANN`, and `OpenCV`. 

For the ONNX inference engine, it supports so many devices (CPUs, GPUs, NPUs) that it can use a multitude of backends, e.g. for NVIDIA GPUs you need to execute the following installation commands:
```
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
You can also install the ONNX runtime for CPU by running:
```
pip install onnxruntime
```
We default to the ONNX runtime for CPU. You can refer more informations from ONNX runtime website: https://onnxruntime.ai/docs/install/.

For the OpenVINO inference engine, you need to install the OpenVINO toolkit first. You can refer more informations from OpenVINO website: https://docs.openvinotoolkit.org/latest/index.html. You can also install the OpenVINO toolkit by running the following command:
```
pip install openvino==2025.1.0
```

For TensorRT inference engine, you need to install the TensorRT Python API first. You can refer more informations from TensorRT website: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html. 

For CoreML inference engine, you need to install the CoreML Tools first. You can refer more informations from CoreML Tools website: https://coremltools.readme.io/docs/installation. You can also install the CoreML Tools by running the following command:
```
pip install coremltools 
```

For CANN inference engine, you need to install CANN first. You can refer more informations from the official website: https://www.hiascend.com/document/. 

For the OpenCV inference engine, you don't need to install anything. We default to the OpenCV inference engine. But our opencv only support CPU. If you want to use GPU or other devices, you need to build OpenCV from source with the corresponding backend.

## Command-line workflows

AnimalPoseTracker provides unified `train`, `val`, `predict`, and `export` commands through `animalpose-cli`. Run them from the project directory; each command accepts the existing project configuration and workflow-specific options:

```bash
animalpose-cli --help
animalpose-cli create --dataset trimouse --workspace projects --name smoke --worker user --model AnimalRTPose --scale N --date demo --epochs 100 --batch-size 8 --image-size 640 --optimizer AdamW --learning-rate 0.001 --no-amp
animalpose-cli load projects/smoke-user-AnimalRTPose-N-demo
animalpose-cli train --config configs/runtime.yaml
animalpose-cli val --config configs/runtime.yaml --weights runs/train/weights/best.pt
animalpose-cli predict --config configs/runtime.yaml --weights runs/train/weights/best.pt --split test --conf 0.25 --iou 0.7 --max-detections 300
animalpose-cli export --config configs/runtime.yaml --weights runs/train/weights/best.pt --format onnx --dynamic --opset 17 --simplify
```

Training options can be overridden for one run without editing `configs/runtime.yaml`:

```bash
animalpose-cli train --config configs/runtime.yaml --epochs 1000 --batch-size 8 --image-size 640 --device 0 --num-workers 8
```

Prediction options such as confidence/IoU thresholds, maximum detections, drawing, tracking and smoothing can also be passed on the command line. Export options such as precision, dynamic input, opset and simplification can be passed the same way. For example:

```bash
animalpose-cli predict --config configs/runtime.yaml --weights runs/train/weights/best.pt --source images/ --conf 0.4 --no-show-boxes --tracker bytetrack --track-track-buffer 45
animalpose-cli export --config configs/runtime.yaml --weights runs/train/weights/best.pt --format onnx --image-size 640 --dynamic --opset 17 --no-simplify
animalpose-cli val --config configs/runtime.yaml --weights runs/train/weights/best.pt --batch-size 8 --image-size 640 --validation-confidence 0.001 --validation-iou 0.7
```

Use `animalpose-cli <command> --help` to see a workflow's options. For train, val, predict and export, command-line values take precedence over `configs/runtime.yaml` for that run and are not written back to the file. Omitted values continue to come from the runtime config or the command's documented default. The `animalpose-cli` commands work without GUI dependencies; install `.[gui]` to use the `animalposetracker` GUI command.

`create` accepts the training settings supported by `train` and saves only explicitly supplied values into the new project's `configs/runtime.yaml`; omitted options retain the selected model's defaults. `load` accepts a project directory, `project.yaml`, or `configs/runtime.yaml`, validates the project and its model/data config paths, and prints a summary without opening the GUI or modifying files.

`animalpose-cli infer` is the deployment-side runtime: it runs an exported artifact (`.onnx`, `.engine`, `.mlpackage`, ...) on a camera, video file, or stream URL, and reads no project configuration at all. Its own yaml is the whole configuration surface — weights, engine and device, source, thresholds, drawing, tracking and filtering, alongside the class/keypoint description (`names` / `kpt_shape` / `skeleton`) that decodes the model's output tensor. Generate the documented template, edit a copy, and point the run at it:

```bash
animalpose-cli infer --write-template                  # writes ./infer.example.yaml
animalpose-cli infer --config infer.yaml               # weights, source, thresholds come from the file
animalpose-cli infer --config infer.yaml --conf 0.4    # an explicit flag wins for that run
```

`--write-template -` prints the template instead of writing it. Values in the file become that run's defaults; a key the file does not define and the data description does not use is reported on stderr and ignored. `--weights` and `--source` have no built-in defaults: either the file or the command line must supply them. Infer refuses `.pt` checkpoints — export first (`animalpose-cli export`), or use `predict` for checkpoints.

`train` exit codes: `0` when training reaches `epochs` **and** when early stopping ends it (an exhausted `early_stopping_patience` is a normal completion — `best.pt` is final), `130` only when the run was interrupted by an external request (`SIGINT`/`SIGTERM` or the `.animalposetracker_training_stop` marker), and `1` on error. Chain `val`/`export` behind `train` in a `set -e` script: early stopping no longer aborts the pipeline.

`early_stopping_patience` is counted in **epochs** since the best validation (`epoch - best_epoch`), so it means the same thing at any `validation_interval`; `0` disables early stopping. The `stopped` event reports `epochs_without_improvement`, `best_epoch`, `patience` and `validation_interval`.

AnimalViTPose validation does **not** use flip test-time augmentation by default, so the reported `coco/AP`, `PCK`, `AUC` and `EPE` match what the shipped decoder (`postprocessing/output_decoders.decode_simcc`) produces. TTA only exists inside the validator — `decode_simcc` receives already-computed output arrays and has no model handle, so it could only ever be a validation-side number. Set `validation.flip_tta: true` in the model preset (`animalposetracker/cfg/training.yaml`) to re-enable it, and expect the metrics to read higher than the deployment path. Any run started before this change that reported numbers with TTA on will therefore look slightly worse afterwards; that is the size of the previous optimism, not a regression.

# Usage: How to use AnimalPoseTracker to train and test a model

1. Open the command prompt or terminal and activate the conda environment:
```
conda activate animalposetracker
```

2. Run the `animalposetracker` command to start the toolbox:
```
animalposetracker
```

3. You can create a new project by user-defined dataset or public dataset, or open an existing project. Click "Create New Project" to create a new project.

https://github.com/user-attachments/assets/156b6e16-5660-4c81-b14b-73120b67417a

4. Click "Public Datasets Project" to create a project based on public datasets, and you need to download datasets.

https://github.com/user-attachments/assets/7c850634-9aef-4b72-95f3-1b11c491a25c

5. Click "Load Project" to load an existing project.

https://github.com/user-attachments/assets/e92ca200-421e-4bce-936e-92cc2d456f12

6. You can Manage Configuration to set the project configuration, and also supports to other configuration settings.

https://github.com/user-attachments/assets/c48b6511-7979-4a84-ac59-b9ba4d6edebe

7. You extract frames from the videos automatically or manually.

https://github.com/user-attachments/assets/61662bd7-4ee4-4266-92f1-6f22bedcac3b

8. You can manually label the animal poses using 'animalpose-annotator' plugin.

https://github.com/user-attachments/assets/6ec6ac35-b6c9-41ee-8107-403861ec054a

https://github.com/user-attachments/assets/3e92d508-c1e1-4dae-8ea7-848c5b9237d8

9. You can train a model using the labeled data and set the training configuration.

https://github.com/user-attachments/assets/79636a6e-3813-40f1-8595-06040e2fe178

10. You can evaluate the trained model on the test set and get the evaluation results.

https://github.com/user-attachments/assets/f352d5fe-1303-4070-b810-b26769faab28

11. You can inference the videos or `test` set using the trained model. If you deploy the model, you should use `animalpose-inferencer` plugin to inference the videos or cameras.

https://github.com/user-attachments/assets/ba5d9c05-73ec-44ae-b4b1-8100f02f1998

12. You can export the trained model to different inference engines.

https://github.com/user-attachments/assets/4abf70c0-e277-4860-ae12-a87e3051a3c8

# Usage: How to use AnimalPoseTracker plugins to inference videos and cameras

1. Open the command prompt or terminal and activate the conda environment:
```
conda activate animalposetracker
```

2. Run the `animalpose-inferencer` command to start the toolbox:
```
animalpose-inferencer
```

3. You can choose the configuration file and the weights file. The software will automatically show the available inference engines and devices.

The configuration file describes the classes and the keypoints. Either style is accepted: a project yaml (`classes_name` / `keypoints_name` / `skeleton`, which is what a project created by this toolbox writes) or a dataset yaml (`names` / `kpt_shape` / `skeleton`). These three keys are not only labels — their counts tell the decoder how the model's output tensor is laid out, so a configuration that disagrees with the artifact shifts the keypoint block and decodes nonsense. Both files are therefore checked as soon as they are picked:

- missing class names or keypoint count: a message explains which key is missing, and Start stays disabled;
- the keypoint count disagrees with the artifact's metadata: a warning is printed and the run is still allowed, because the decoder follows the configuration;
- the artifact holds an AnimalViTPose SimCC model: refused while selecting, since this toolbox infers YOLO-Pose layouts only. Run those through `animalposetracker predict`.

https://github.com/user-attachments/assets/db726dd8-a092-40e3-94ea-e3d423973a03

4. You can enable the camera or video to inference. Then, you can choose a video from the local directory or a camera to inference. And, you can preview video or camera.

https://github.com/user-attachments/assets/007caeb6-b5b2-48a8-9873-58ff9db002b7

5. You can set camera parameters. Note: It requires the camera to support the specified resolution and FPS, set Model Bits, but it requires you know the model precision, and set the inference engine. It will automatically show the available inference devices, and set the inference device.

https://github.com/user-attachments/assets/e67da907-e765-4247-affb-b85563a33d28

6. You can start the inference. And, you can set many parameters to control the inference process, such as the confidence threshold, and the visualization style.

https://github.com/user-attachments/assets/0b57bba6-9ed6-47b1-9f4f-1d449762c784

https://github.com/user-attachments/assets/9d2ad685-7201-4488-a653-500a6c72b3bf

7. You can turn tracking on. Tick `Track` and pick one of the tracker algorithms; every label then carries its track id (`#1 mouse: 0.90`). The list is the same one `infer --tracker` accepts, and a tracker follows the pose records rather than the pixels — which is why the same run can be replayed later with another algorithm. `Track`, `Filter` and the `Conf`/`IoU` thresholds sit together under the file pickers, above the source, engine and device choices: all of them decide what a run produces rather than what is drawn.

8. The per-frame pose records are saved with every run, into `runs/inference/records.json`, in the same `[{frame, predictions}, ...]` shape `infer --records` writes, with a `track_id` on every detection when tracking was on. There is no switch for it: the records are the run's result, and they are the file tracking workbenches, trajectory tools and 3D reconstruction read. `Save Video` is the one switch here — it writes the rendered frames to `runs/inference/output.avi`.

9. You can smooth the tracked poses over time by ticking `Filter` and choosing a method (`median` or `savgol`) with its window, polynomial order, keypoint confidence and gap settings. The smoothing runs over the whole sequence when the run ends, so it cannot affect what you see live — the records file holds the smoothed records. Filtering needs track ids, so the filter controls stay disabled until `Track` is ticked, exactly as `--pose-filter` requires `--tracker` on the command line.

# Usage: How to use AnimalPoseTracker plugins to label and annotate animal poses

1. Open the command prompt or terminal and activate the conda environment:
```
conda activate animalposetracker
```

2. Run the `animalpose-annotator` command to start the toolbox:
```
animalpose-annotator
```

3. You can choose the configuration file. You can choose the images directory.

https://github.com/user-attachments/assets/7d820fbe-390e-4387-9678-ac229e930cda

4. You can draw 'bounding box'. You can choose the object class. You can draw 'keypoints'. You can set the keypoint class. You can draw 'skeleton'. You can save the annotations..You can export the annotations to a file.

https://github.com/user-attachments/assets/a22b28df-4b12-48b0-a087-4ab07f6ea17f

# Citation: How to cite AnimalPoseTracker

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
