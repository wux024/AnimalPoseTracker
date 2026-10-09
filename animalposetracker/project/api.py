"""Public project API composed from focused project service components."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS, DEFAULT_CFG_PATH
from .configuration import ProjectConfigurationMixin
from .files import ProjectFilesMixin
from .weights import ProjectWeightsMixin, _download_google_drive_checkpoint
from .jobs import ProjectJobsMixin

class AnimalPoseTrackerProject(ProjectConfigurationMixin, ProjectFilesMixin, ProjectWeightsMixin, ProjectJobsMixin):
    CONFIG_TYPES = {"project", "dataset", "model", "other"}

    DEFAULT_DIRS = [
        "",
        "configs",
        "datasets",
        "datasets/images",
        "datasets/images/train",
        "datasets/images/val",
        "datasets/images/test",
        "datasets/labels",
        "datasets/labels/train",
        "datasets/labels/val",
        "datasets/labels/test",
        "datasets/annotations",
        "pretrained",
        "runs",
        "sources",
        "sources/images",
        "sources/videos",
        "sources/extracted",
        "sources/extracted/yolo_format",
        "sources/extracted/coco_format",
    ]

    def __init__(self, local_path: Union[str, Path] = None,
                 project_name: str = 'person',
                 worker: str = 'Adam',
                 model_type: str = 'AnimalRTPose',
                 model_scale: str = 'N',
                 keypoints: int = 17,
                 visible: bool = True,
                 classes: int = 1,
                 keypoints_name: Optional[List[str]] = None,
                 skeleton: Optional[List[List[int]]] = None,
                 kpt_oks_sigmas: Optional[List[float]] = None,
                 classes_name: Optional[List[str]] = None,
                 sources: Union[str, Path, List[Union[str, Path]]] = None,
                 date: str = None):
        """
        Initialize a new project with given parameters.

        Args:
            local_path: Path to project directory (default: current directory)
            project_name: Name of the project (default: 'person')
            worker: Name of worker creating project (default: 'Adam')
            model_type: Type of model to use: 'AnimalRTPose')
            model_scale: Scale of model (default: 'N')
            keypoints: Number of keypoints (default: 17)
            visible: Whether keypoints have visibility info (default: True)
            classes: Number of classes (default: 1)
            keypoints_name: Names of keypoints (default: auto-generated)
            skeleton: connections between keypoints (default: empty)
            kpt_oks_sigmas: Per-keypoint OKS sigmas (default: uniform distribution)
            classes_name: Names of classes (default: ['person'])
            sources: Source paths (default: None)
            date: Project creation date (default: current date)
        """
        # Validate keypoints-related parameters
        if keypoints_name is not None and len(keypoints_name) != keypoints:
            raise ValueError(
                f"keypoints_name length ({len(keypoints_name)}) must match keypoints ({keypoints})"
            )

        if skeleton is not None:
            for connection in skeleton:
                if any(idx >= keypoints for idx in connection):
                    raise ValueError(
                        f"Skeleton contains invalid keypoint index (max {keypoints-1})"
                    )

        # Initialize paths and basic attributes
        kpt_shape = [keypoints, 3] if visible else [keypoints, 2]

        # Handle default values
        if classes_name is None:
            classes_name = ['person']

        if kpt_oks_sigmas is None:
            kpt_oks_sigmas = [1.0 / keypoints] * keypoints  # Default uniform distribution

        self.project_config = {
            "project_path": None,
            "project_name": project_name,
            "worker": worker,
            "model_type": model_type,
            "model_scale": model_scale,
            "date": date if date is not None else datetime.now().strftime(r"%Y%m%d"),
            "keypoints": keypoints,
            "visible": visible,
            "classes": classes,
            "keypoints_name": keypoints_name or [f"kpt_{i}" for i in range(keypoints)],
            "skeleton": skeleton or [],
            "classes_name": classes_name,
            "sources": [str(s) for s in sources] if sources else [],
        }

        self.dataset_config = {
            'path': 'datasets',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'kpt_shape': kpt_shape,
            'flip_idx': list(range(keypoints)),
            'names': dict(enumerate(classes_name)),
            'skeleton': skeleton or [],
            'kpt_oks_sigmas': kpt_oks_sigmas,
        }

        self.model_config = {
            'nc': classes,
            'kpt_shape': kpt_shape,
            'scales': None,
            'backbone': None,
            'head': None,
        }

        # include trian, val and inference config here
        self.other_config = {}
        self._other_training_tree = None

        # Generate project path and initialize configurations
        self.local_path = Path(local_path) if local_path else Path.cwd()
        self.project_config['project_path'] = str(self._project_path)

        # process
        self.process = None
        self.training_event_path = None
        self._training_event_offset = 0
        self.prediction_output_dir = None

    def _download_pretrained_checkpoint(self, url, destination):
        """Route downloads through the project module's patchable public seam."""
        return _download_google_drive_checkpoint(url, destination)


__all__ = ["AnimalPoseTrackerProject", "_download_google_drive_checkpoint"]
