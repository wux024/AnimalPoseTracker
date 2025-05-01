import cv2


PLATFORM = {
    "Intel": ["Intel CPU", "Intel GPU", "Intel NPU", "NVIDIA GPU", "NVIDIA GPU TensorRT"],
    "AMD": ["AMD CPU", "AMD GPU", "NVIDIA GPU", "NVIDIA GPU TensorRT"],
    "ARM": ["ARM CPU", "Metal", "NVIDIA GPU", "NVIDIA GPU TensorRT", "Ascend NPU"],
}

ENGINEtoDEVICE = {
    "OpenCV": {
        "Intel": ["Intel CPU", "Intel GPU", "Intel NPU", "NVIDIA GPU"],
        "AMD": ["AMD CPU", "AMD GPU", "NVIDIA GPU"],
        "ARM": ["ARM CPU", "Metal", "NVIDIA GPU", "Ascend NPU"],
    },
    "OpenVINO": 
    {
        "Intel": ["Intel CPU", "Intel GPU", "Intel NPU"],
        "AMD": [],
        "ARM": [],
    },
    "CANN":
    {
        "Intel": [],
        "AMD": [],
        "ARM": ["Ascend NPU"],
    },
    "ONNX": PLATFORM,
    "TensorRT": 
    {
        "Intel": ["NVIDIA GPU TensorRT"],
        "AMD": ["NVIDIA GPU TensorRT"],
        "ARM": ["NVIDIA GPU TensorRT"],
    },
    "CoreML": 
    {
        "Intel": [],
        "AMD": [],
        "ARM": ["Metal"]
    }
}


ENGINEtoBackend = {
    "OpenCV": 
    {
        "CPU": cv2.dnn.DNN_BACKEND_OPENCV,
        "Metal": cv2.dnn.DNN_BACKEND_OPENCV,
        "NVIDIA GPU": cv2.dnn.DNN_BACKEND_CUDA,
        "Intel GPU": cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        "Intel NPU": cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        "Ascend NPU": cv2.dnn.DNN_BACKEND_CANN,
    },
    "OpenVINO": 
    {
        "Intel CPU": "CPU",
        "Intel GPU": "GPU",
        "Intel NPU": "NPU",
    },
    "ONNX": 
    {
        "Intel CPU": ['DnnlExecutionProvider',
                      'OpenVINOExecutionProvider', 'CPUExecutionProvider'],
        "Intel NPU": ['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
        "Intel GPU": ['DmlExecutionProvider', 
                      'OpenVINOExecutionProvider', 'CPUExecutionProvider'],
        "AMD CPU": ['DnnlExecutionProvider', 'CPUExecutionProvider'],
        "AMD GPU": ['MIGraphXExecutionProvider', 
                    'ROCMExecutionProvider', 'CPUExecutionProvider'],
        "ARM CPU": ['ArmNNExecutionProvider', 
                    'ACLExecutionProvider', 'CPUExecutionProvider'],
        "Metal": ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
        "Ascend NPU": ['CANNExecutionProvider', 'CPUExecutionProvider'],
        "NVIDIA GPU": ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        "NVIDIA GPU TensorRT": ['TensorrtExecutionProvider', 
                       'CUDAExecutionProvider', 'CPUExecutionProvider'],
    },
}

OpenCV_TARGETS = {
    "CPU": cv2.dnn.DNN_TARGET_CPU,
    "CPU FP16": cv2.dnn.DNN_TARGET_CPU_FP16,
    "NVIDIA GPU": cv2.dnn.DNN_TARGET_CUDA,
    "NVIDIA GPU FP16": cv2.dnn.DNN_TARGET_CUDA_FP16,
    "Intel GPU": cv2.dnn.DNN_TARGET_OPENCL,
    "Intel GPU FP16": cv2.dnn.DNN_TARGET_OPENCL_FP16,
    "Intel NPU": cv2.dnn.DNN_TARGET_NPU,
    "Ascend NPU": cv2.dnn.DNN_TARGET_NPU,
    "Metal": cv2.dnn.DNN_TARGET_OPENCL,
    "Metal FP16": cv2.dnn.DNN_TARGET_OPENCL_FP16,
}

EP_PARAMS = {
    # ==================== CPU Providers ====================
    "CPUExecutionProvider": {
        "use_arena": {
            "type": int,
            "default": 1,
            "options": [0, 1],
            "description": "Enable memory arena optimization (1=enabled, 0=disabled)"
        },
        "arena_extend_strategy": {
            "type": int,
            "default": 0,
            "options": [0, 1],
            "description": "Memory allocation strategy (0=kSameAsRequested, 1=kNextPowerOfTwo)"
        },
        "enable_mem_pattern": {
            "type": bool,
            "default": True,
            "description": "Enable memory pattern reuse optimization"
        },
        "thread_pool_size": {
            "type": int,
            "default": None,
            "description": "Number of threads in thread pool (None=auto-detect)"
        }
    },

    "DnnlExecutionProvider": {
        "use_arena": {
            "type": int,
            "default": 1,
            "description": "Enable memory arena for oneDNN"
        },
        "thread_pool_size": {
            "type": int,
            "default": None,
            "description": "Thread pool size for oneDNN"
        },
        "use_dnnl_heuristic": {
            "type": bool,
            "default": True,
            "description": "Enable oneDNN heuristic algorithm selection"
        }
    },

    # ==================== NVIDIA GPU Providers ====================
    "CUDAExecutionProvider": {
        "device_id": {
            "type": int,
            "default": 0,
            "description": "GPU device ID to use"
        },
        "cudnn_conv_algo_search": {
            "type": int,
            "default": 0,
            "options": [0, 1],
            "description": "cuDNN convolution algorithm search (0=DEFAULT, 1=EXHAUSTIVE)"
        },
        "do_copy_in_default_stream": {
            "type": bool,
            "default": True,
            "description": "Perform copies in default stream"
        },
        "enable_cuda_graph": {
            "type": bool,
            "default": False,
            "description": "Enable CUDA graph capture"
        }
    },

    "TensorrtExecutionProvider": {
        "device_id": {
            "type": int,
            "default": 0,
            "description": "GPU device ID"
        },
        "trt_fp16_enable": {
            "type": bool,
            "default": False,
            "description": "Enable FP16 precision"
        },
        "trt_int8_enable": {
            "type": bool,
            "default": False,
            "description": "Enable INT8 quantization"
        },
        "trt_engine_cache_enable": {
            "type": bool,
            "default": False,
            "description": "Enable engine caching"
        },
        "trt_engine_cache_path": {
            "type": str,
            "default": "",
            "description": "Path to save/load TRT engine cache"
        },
        "trt_max_workspace_size": {
            "type": int,
            "default": 1 << 30,  # 1GB
            "description": "Max workspace size in bytes"
        }
    },

    # ==================== Other GPU Providers ====================
    "DmlExecutionProvider": {
        "device_id": {
            "type": int,
            "default": 0,
            "description": "DirectML device ID"
        },
        "use_dedicated_allocator": {
            "type": bool,
            "default": True,
            "description": "Use dedicated memory allocator"
        }
    },

    "ROCMExecutionProvider": {
        "device_id": {
            "type": int,
            "default": 0,
            "description": "AMD GPU device ID"
        },
        "enable_hip_graph": {
            "type": bool,
            "default": False,
            "description": "Enable HIP graph optimization"
        }
    },

    # ==================== Specialized Hardware ====================
    "OpenVINOExecutionProvider": {
        "device_type": {
            "type": str,
            "default": "CPU",  # Updated from CPU_FP32
            "options": ["CPU", "GPU", "GPU.0", "GPU.1", "NPU", "HETERO:CPU,GPU", "MULTI:CPU,GPU", "AUTO"],
            "description": "Target hardware type (CPU_FP32 deprecated)"
        },
        "num_of_threads": {
            "type": int,
            "default": 8,  # Changed from None to a valid number
            "description": "Number of threads to use (must be integer)"
        },
        "precision": {
            "type": str,
            "default": "FP32",
            "options": ["FP32", "FP16", "INT8"],
            "description": "Precision mode (separate from device_type now)"
        }
    },
    "CoreMLExecutionProvider": {
        "coreml_flags": {
            "type": int,
            "default": 0,
            "options": [0, 1, 2],
            "description": "Execution flags (0=auto, 1=CPU only, 2=use ANE)"
        }
    },
    "CANNExecutionProvider": {
        "device_id": {
            "type": int,
            "default": 0,
            "description": "Ascend NPU device ID"
        },
        "precision_mode": {
            "type": str,
            "default": "force_fp16",
            "description": "Precision mode for NPU"
        }
    }
}