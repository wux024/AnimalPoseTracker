"""Public project API composed from focused project service components."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union

from .configuration import ProjectConfigurationMixin
from .files import ProjectFilesMixin
from .weights import ProjectWeightsMixin, _download_google_drive_checkpoint
from .jobs import ProjectJobsMixin

class AnimalPoseTrackerProject(ProjectConfigurationMixin, ProjectFilesMixin, ProjectWeightsMixin, ProjectJobsMixin):
    CONFIG_TYPES = {"project", "dataset", "model", "runtime"}

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
        "results",
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
                 date: str = None,
                 pose3d: bool = False,
                 public_dataset: Optional[str] = None):
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
            pose3d: Enable multi-view 3D reconstruction (default: False)
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

        # The dataset mapping must cover exactly the classes the model outputs;
        # a caller can hold fewer name rows than the count asks for.
        classes_name = [
            *list(classes_name)[:classes],
            *[f"class{index + 1}" for index in range(len(classes_name), classes)],
        ]

        self.project_config = {
            "project_path": None,
            "project_name": project_name,
            "worker": worker,
            "model_type": model_type,
            "model_scale": model_scale,
            "date": date if date is not None else datetime.now().strftime(r"%Y%m%d"),
            "sources": [str(s) for s in sources] if sources else [],
            # Multi-view reconstruction is opt-in: only a project created with the 3D
            # option carries the calibration file and results/pose3d.
            "pose3d": bool(pose3d),
            "public_dataset": public_dataset or None,
        }

        self.pose3d = bool(pose3d)

        self.dataset_config = {
            'path': 'datasets',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'kpt_shape': kpt_shape,
            # No left/right table yet: a keypoint count says nothing about which points mirror
            # onto which, and a made-up identity table would mirror the coordinates without
            # swapping the labels. Training reads the absence and leaves flips off until a real
            # table arrives (a descriptor, or an edit to dataset.yaml).
            'flip_idx': None,
            'names': dict(enumerate(classes_name)),
            'skeleton': skeleton or [],
            'kpt_oks_sigmas': kpt_oks_sigmas,
        }
        if keypoints_name is not None:
            # The dataset file is the keypoint names' home now; the project
            # file keeps identity only.
            self.dataset_config["keypoints_name"] = [str(name) for name in keypoints_name]

        self.model_config = {
            'nc': classes,
            'kpt_shape': kpt_shape,
            'scales': None,
            'backbone': None,
            'head': None,
        }

        # The project's runtime configuration (configs/runtime.yaml): filled
        # by create_runtime_config at creation, or by load_config_file.
        self.runtime_config = {}

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
