from typing import Dict, Union
import cv2
import numpy as np
import yaml
from pathlib import Path
import os

from animalposetracker.utils.base import measure_time

from .constant import ENGINEtoBackend, OpenCV_TARGETS, EP_PARAMS
from .inferencer import InferenceEngine

class TrackerEngine(InferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)