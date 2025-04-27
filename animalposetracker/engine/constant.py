import cv2


PLATFORM = {
    "Intel": ["Intel CPU", "Intel GPU", "Intel NPU", "NVIDIA GPU"],
    "AMD": ["AMD CPU", "AMD GPU", "NVIDIA GPU"],
    "ARM": ["ARM CPU", "Metal", "NVIDIA GPU", "Ascend NPU"],
}

ENGINEtoDEVICE = {
    "OpenCV": PLATFORM,
    "OpenVINO": 
    {
        "Intel": ["Intel CPU", "Intel GPU", "Intel NPU"],
    },
    "CANN":
    {
        "ARM": ["Ascend NPU"],
    },
    "ONNX": PLATFORM,
    "TensorRT": 
    {
        "Intel": ["NVIDIA GPU"],
        "AMD": ["NVIDIA GPU"],
        "ARM": ["NVIDIA GPU"],
    },
    "CoreML": 
    {
        "ARM": ["Metal"]
    }
}


ENGINEtoBackend = {
    "OpenCV": 
    {
        "CPU": cv2.dnn.DNN_BACKEND_OPENCV,
        "NVIDIA GPU": cv2.dnn.DNN_BACKEND_CUDA,
        "Intel GPU": cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        "Intel NPU": cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        "Ascend NPU": cv2.dnn.DNN_BACKEND_CANN,
    },
    "OpenVINO": 
    {
        "Intel CPU": "CPU",
        "Intel GPU": "GPU",
        "Intel NPU": "MYRIAD",
    },
    "ONNX": 
    {
        "CPU": ["CPUExecutionProvider"],
        "NVIDIA GPU": ['CUDAExecutionProvider'],
        "NVIDIA GPU TensorRT": ['TensorrtExecutionProvider', 'CUDAExecutionProvider'],
        "Intel NPU": ['OpenVINOExecutionProvider'],
        "Intel GPU": ['OpenVINOExecutionProvider'],
        "AMD GPU": ['ROCMExecutionProvider'],
        "Metal": ['CoreMLExecutionProvider'],
        "Ascend NPU": ['AscendExecutionProvider'],
    },
}

OpenCV_TARGETS = {
    "CPU": cv2.dnn.DNN_TARGET_CPU,
    "CPU FP16": cv2.dnn.DNN_TARGET_OPENCL_FP16,
    "NVIDIA GPU": cv2.dnn.DNN_TARGET_CUDA,
    "NVIDIA GPU FP16": cv2.dnn.DNN_TARGET_CUDA_FP16,
    "Intel GPU": cv2.dnn.DNN_TARGET_MYRIAD,
    "Intel NPU": cv2.dnn.DNN_TARGET_MYRIAD,
    "Ascend NPU": cv2.dnn.DNN_TARGET_NPU,
}