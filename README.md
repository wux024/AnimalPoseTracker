![](https://s3.bmp.ovh/imgs/2025/05/15/e246d2b0dec75c56.png)

# Welcome! 👋

**AnimalPoseTracker™️** is a toolbox for cross-species animal pose estimation. 

# Installation: How to install AnimalPoseTracker

1. Create a conda environment:
We recommend creating a new conda environment for AnimalPoseTracker, recommended Python >= 3.8. You can do this by running the following command in your terminal:
```
conda create -n animalposetracker python=3.8
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
