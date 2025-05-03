from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional
from PySide6.QtCore import QThread, Signal, QMutex

@dataclass
class FrameInfo:
    """
    Base container class for video frame data and processing results.
    
    Attributes:
        original_frame (Optional[np.ndarray]): Raw captured frame in BGR format
        preprocess_frame (Optional[np.ndarray]): Preprocessed input tensor for model inference
        IM (Optional[np.ndarray]): Affine transformation matrix for coordinate mapping
        results (Dict): Dictionary containing detection results and timing metrics:
            - boxes: Bounding box coordinates (xyxy format)
            - keypoints_list: Detected keypoints array
            - class_ids: Class identification numbers
            - scores: Confidence scores [0-1]
            - preprocess_time: Time taken for preprocessing (ms)
            - inference_time: Model inference latency (ms)
            - postprocess_time: Result parsing time (ms)
            - fps: Estimated pipeline throughput
    """
    end: bool = False
    original_frame: Optional[np.ndarray] = None
    preprocess_frame: Optional[np.ndarray] = None
    IM: Optional[np.ndarray] = None
    results: Dict = field(default_factory=lambda: {
        'pred' : None,
        'boxes': None,
        'keypoints_list': None,
        'class_ids': None,
        'scores': None,
        'preprocess_time': 0.0,
        'inference_time': 0.0,
        'postprocess_time': 0.0,
        'fps': 0.0
    })


class BaseThread(QThread):
    """
    Base thread class that provides common attributes and methods.
    """
    status_update = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self._lock = QMutex()

    @property
    def is_running(self):
        self._lock.lock()
        is_running = self.running
        self._lock.unlock()
        return is_running

    def safe_stop(self):
        self._lock.lock()
        self.running = False
        self._lock.unlock()
        self.wait()

    def _release_resources(self):
        pass