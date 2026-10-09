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

3. Install AnimalPoseTracker with its optional training dependencies:
Clone the repository:
```
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
pip install -v -e ".[training]"
```
or 
```
pip install git+https://github.com/wux024/AnimalPoseTracker.git
```

Project training, validation, prediction, and model export use AnimalPoseTracker's own workflows; they do not require the YOLO training framework or CLI. Install `.[training]` for PyTorch and COCO evaluation support. Export targets include TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow SavedModel/GraphDef/Lite/Edge TPU/JS, PaddlePaddle, MNN, NCNN, IMX500, and RKNN. Project prediction also accepts a remote Triton model URL; TensorFlow.js is export-only, matching the Ultralytics AutoBackend baseline. Each target needs its optional runtime or vendor SDK; see [deployment format support](docs/deployment_formats.md) for the matrix and installation extras.

AnimalViTPose uses the same built-in trainer as AnimalRTPose, with a top-down instance-crop data adapter and the SimCC/KL loss, optimizer, schedule, and COCO/PCK/AUC/EPE evaluation settings from the [MMPose AnimalViTPose recipe](https://github.com/wux024/mmpose/blob/main/configs/animal_2d_keypoint/animalvitpose/ap10k/animalvitpose-small_8xb64-210e_ap10k-256x256.py). The MMPose Small/Base/Large/Huge scales and project image size select the ViT graph inside the same trainer. MMPose itself is not imported at training time; install `.[training]` for PyTorch and COCO evaluation support.

Built-in training defaults are defined in [`animalposetracker/cfg/training.yaml`](animalposetracker/cfg/training.yaml). Values identical across AnimalRTPose and AnimalViTPose are defined once under `training.shared`; each model profile contains only its specific or differing training, loss, augmentation and validation values. The model type in `project.yaml` selects the matching key under `training.models`, so the selection is stored only once. Network graphs, AnimalViTPose scale variants, SimCC head geometry and input normalization stay in the single selected project's `configs/model.yaml`; dataset paths and keypoint metadata stay in `configs/dataset.yaml`. The selected scale comes from `project.yaml` and is applied to the model graph at training time. Project code exposes the selected training profile as a flat view to the existing settings UI.

`project.yaml` also keeps GUI-facing class/keypoint names, counts and skeleton metadata because existing annotation pages read that file directly; `dataset.yaml` carries the corresponding loader fields. `model.yaml` repeats `kpt_shape` as the model output contract, which must match the dataset. OKS sigmas have a single owner: `dataset.yaml`.

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
animalpose-cli train --config configs/other.yaml
animalpose-cli val --config configs/other.yaml --weights runs/train/weights/best.pt
animalpose-cli predict --config configs/other.yaml --weights runs/train/weights/best.pt --split test
animalpose-cli export --config configs/other.yaml --weights runs/train/weights/best.pt --format onnx
```

Use `animalpose-cli <command> --help` to see a workflow's options. The existing `animalposetracker` command continues to launch the GUI.

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

https://github.com/user-attachments/assets/db726dd8-a092-40e3-94ea-e3d423973a03

4. You can enable the camera or video to inference. Then, you can choose a video from the local directory or a camera to inference. And, you can preview video or camera.

https://github.com/user-attachments/assets/007caeb6-b5b2-48a8-9873-58ff9db002b7

5. You can set camera parameters. Note: It requires the camera to support the specified resolution and FPS, set Model Bits, but it requires you know the model precision, and set the inference engine. It will automatically show the available inference devices, and set the inference device.

https://github.com/user-attachments/assets/e67da907-e765-4247-affb-b85563a33d28

6. You can start the inference. And, you can set many parameters to control the inference process, such as the confidence threshold, and the visualization style.

https://github.com/user-attachments/assets/0b57bba6-9ed6-47b1-9f4f-1d449762c784

https://github.com/user-attachments/assets/9d2ad685-7201-4488-a653-500a6c72b3bf

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
