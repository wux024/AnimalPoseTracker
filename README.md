![](https://s3.bmp.ovh/imgs/2025/05/15/e246d2b0dec75c56.png)

# Welcome! 👋

**AnimalPoseTracker™️** is a toolbox for cross-species animal pose estimation. 

# Installation: How to install AnimalPoseTracker

1. Create a conda environment:
We recommend creating a new conda environment for AnimalPoseTracker, recommended Python >= 3.10. You can do this by running the following command in your terminal:
```
conda create -n animalposetracker python=3.10
```

2. [Optional] If you want to train a model, you need to install "PyTorch" and "torchvision" first. You can install them by following the instructions on the official website: https://pytorch.org/get-started/locally/. We recommend Pytorch >= 1.8. For example, if you are using conda, you can run the following command:
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

And then install 'ultralytics' (ours customized) by running the following command:
```
pip install git+https://github.com/wux024/ultralytics.git@animalrtpose
```

3. Install AnimalPoseTracker: 
Clone the repository:
```
git clone https://github.com/wux024/AnimalPoseTracker.git
cd AnimalPoseTracker
pip install -v -e.
```
or 
```
pip install git+https://github.com/wux024/AnimalPoseTracker.git
```

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
