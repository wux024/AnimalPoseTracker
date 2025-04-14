from ultralytics import YOLO
from typing import Dict, List, Tuple
import cv2
import numpy as np
class InferenceEngine:
    ENGINE = ["OpenCV", "OpenVINO", "Ultralytics", "MMdeploy", "CANN", "Hailo"]
    DEVICE = ["CPU", "NVIDIA GPU", "Intel NPU", "Ascend NPU", "Hailo-8"]
    def __init__(self, 
                 data_config: str, 
                 weights_path: str, 
                 engine: str = 'Ultralytics',
                 device: str = 'CPU',
                 save: bool = False,
                 show: bool = False,
                 background: bool = 'Original',
                 show_classes: bool = False,
                 show_keypoints: bool = True,
                 show_skeletons: bool = False,
                 show_bbox: bool = True,
                 radius: int = 5,
                 skeleton_line_width: int = 2,
                 bbox_line_width: int = 2,
                 **kwargs
                 ):
        self.model = None
        self.data_config = data_config
        self.weights_path = weights_path
        self.engine = engine
        self.device = device
        self.visualize_config = {
            'save': save,
            'show': show,
            'show_classes': show_classes,
            'show_keypoints': show_keypoints,
            'show_skeletons': show_skeletons,
            'show_bbox': show_bbox,
            'radius': radius,
            'skeleton_line_width': skeleton_line_width,
            'bbox_line_width': bbox_line_width,
            'background': background
        }
        self.other_config = kwargs

    def engine_selection(self):
        if self.engine not in self.ENGINE:
            raise ValueError(f"Engine {self.engine} is not supported. Please choose from {self.ENGINE}")
        if self.device not in self.DEVICE:
            raise ValueError(f"Device {self.device} is not supported. Please choose from {self.DEVICE}")
        
        if self.engine == 'Ultralytics':
            self.model = YOLO(data=self.data_config, weights=self.weights_path, device=self.device)
        elif self.engine == 'OpenCV':
            pass
        elif self.engine == 'OpenVINO':
            pass
        elif self.engine == 'MMdeploy':
            pass
        elif self.engine == 'CANN':
            pass
        elif self.engine == 'Hailo':
            pass
    
    def infer(self, img):
        return self.model(img)

    def visualize(self, img, results):
        pass
            