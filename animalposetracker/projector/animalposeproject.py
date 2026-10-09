import yaml
import hashlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json
import shutil
import os
import tempfile
from html.parser import HTMLParser

import subprocess
import sys

from animalposetracker.cfg import (
    DATA_YAML_PATHS,
    MODEL_YAML_PATHS,
    DEFAULT_CFG_PATH,
)


class _GoogleDriveConfirmationParser(HTMLParser):
    """Read the hidden confirmation form Google Drive shows for larger files."""

    def __init__(self):
        super().__init__()
        self.forms = []
        self._current_form = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "form":
            self._current_form = {
                "action": attributes.get("action"),
                "params": {},
            }
            self.forms.append(self._current_form)
        elif tag == "input" and self._current_form is not None:
            name = attributes.get("name")
            if name:
                self._current_form["params"][name] = attributes.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self._current_form = None


def _download_google_drive_checkpoint(url, destination) -> None:
    """Download a shared Google Drive checkpoint, including its virus-scan confirmation step."""
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = None
    temporary = None
    try:
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", "").lower():
            parser = _GoogleDriveConfirmationParser()
            parser.feed(response.text)
            response.close()
            form = next(
                (
                    candidate for candidate in parser.forms
                    if candidate.get("action") and candidate.get("params")
                ),
                None,
            )
            if form is None:
                raise RuntimeError(
                    "Google Drive returned an HTML confirmation page without a download form"
                )
            response = session.get(
                form["action"],
                params=form["params"],
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise RuntimeError("Google Drive returned a confirmation page instead of checkpoint data")

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        chunks = (chunk for chunk in response.iter_content(chunk_size=1024 * 1024) if chunk)
        first_chunk = next(chunks, b"")
        if not first_chunk.startswith(b"PK\x03\x04"):
            raise RuntimeError("Google Drive response is not a PyTorch ZIP checkpoint")
        with temporary.open("wb") as stream:
            stream.write(first_chunk)
            for chunk in chunks:
                stream.write(chunk)
        os.replace(temporary, destination)
    finally:
        if response is not None:
            response.close()
        session.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)



class AnimalPoseTrackerProject:
    """Manage animal pose tracking projects including configurations and directory structure."""
    
    # Constants for configuration types
    CONFIG_TYPES = {"project", "dataset", "model", "other"}
    
    # Standard directory structure
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
    
    def _validate_keypoints_params(
        self,
        keypoints: int,
        keypoints_name: Optional[List[str]],
        skeleton: Optional[List[List[int]]]
    ) -> None:
        """Validate keypoints-related parameters."""
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
    
    def _init_config(self, default: Dict[str, Any]) -> Dict[str, Any]:
        return default.copy()

    @property
    def local_path(self) -> Path:
        """Get the local path of the project."""
        return self._local_path

    @local_path.setter
    def local_path(self, value: Union[str, Path]) -> None:
        """Set the local path of the project."""
        self._local_path = Path(value)
        self._project_path = self._generate_project_path()
    
    @property
    def project_path(self) -> Path:
        """Get the project path."""
        return self._project_path

    def _generate_project_path(self) -> Path:
        """Generate the project path based on naming convention."""
        config = self.project_config
        components = [
            config['project_name'],
            config['worker'],
            config['model_type'],
            config['model_scale'],
            str(config['date'])
        ]
        return self.local_path / "-".join(components)

    def print_project_info(self) -> None:
        """Print the current project configuration in a readable format."""
        if not self.project_config:
            print("Project config has not been created.")
            return
            
        print("\nProject Configuration:")
        print("-" * 40)
        for key, value in self.project_config.items():
            print(f"{key:<20}: {value}")
        print("-" * 40)
        

    def create_new_project(self) -> None:
        """Create a new project with default configurations."""
        self._create_project_structure()
        self.load_config_file()
    
    def _create_project_structure(self) -> None:
        """Create all project directories and configurations."""
        self.create_project_dirs()
        for config_type in self.CONFIG_TYPES:
            getattr(self, f"create_{config_type}_config")()
        
    def load_config_file(self) -> None:
        """Load the project configuration from file."""
        config_files = {
            "project": self._project_path / "project.yaml",
            "dataset": self._project_path / "configs" / "dataset.yaml",
            "model": self._project_path / "configs" / "model.yaml",
            "other": self._project_path / "configs" / "other.yaml",
        }
        
        for config_type, config_file in config_files.items():
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            self._load_config_file(config_type, config_file)

    def create_public_dataset_project(self, dataname: str = 'AP10K') -> None:
        """Create a project using a public dataset configuration."""
        if dataname not in DATA_YAML_PATHS:
            raise ValueError(f"Invalid dataset name. Available: {list(DATA_YAML_PATHS.keys())}")
        
        self._create_project_structure()
        self._load_config_from_file("dataset", DATA_YAML_PATHS[dataname])
        
        # Update project config based on dataset
        self.project_config.update({
            "classes": len(self.dataset_config['names'].keys()),
            "classes_name": list(self.dataset_config['names'].values()),
            "keypoints": self.dataset_config['kpt_shape'][0],
            "keypoints_name": [f"kpt_{i}" for i in range(self.dataset_config['kpt_shape'][0])],
            "skeleton": self.dataset_config['skeleton'],
            "visible": self.dataset_config['kpt_shape'][1] == 3
        })
        
        self._update_configs_from_dataset()
        self.load_config_file()
    
    def _update_configs_from_dataset(self) -> None:
        """Update configurations based on dataset settings."""
        self.update_config("project", self.dataset_config)
        self.update_config("project", {"project_path": str(self._project_path)})
        self.update_config("dataset", {'path': str(self._project_path / "datasets")})
        self._sync_model_config_from_dataset()
        self.other_config.update({
            'model': str(self._project_path / "configs" / "model.yaml"),
            'data': str(self._project_path / "configs" / "dataset.yaml")
        })
        self.save_configs()

    def _sync_model_config_from_dataset(self) -> None:
        """Keep the model YAML's output shape aligned with the project dataset."""
        model_nc = (
            1 if self.project_config.get("model_type") == "AnimalViTPose"
            else self.project_config["classes"]
        )
        self.model_config.update({
            'nc': model_nc,
            'kpt_shape': list(self.dataset_config['kpt_shape']),
        })
        if self.project_config.get("model_type") != "AnimalViTPose":
            return

        head_entries = self.model_config.get("head") or []
        if not head_entries or len(head_entries[-1]) < 4 or head_entries[-1][2] != "SimCCHead":
            raise ValueError("AnimalViTPose model configuration must end with a SimCCHead")
        head_args = head_entries[-1][3]
        if not isinstance(head_args, list) or not head_args:
            raise ValueError("AnimalViTPose SimCCHead configuration has no keypoint count")
        if head_args[0] != "num_keypoints":
            head_args[0] = int(self.dataset_config['kpt_shape'][0])
    
    def load_project_config(self, config_path: Union[str, Path]) -> None:
        """Load a project configuration from file."""
        self._load_config_from_file("project", config_path)
        self.local_path = Path(config_path).parent.parent
        self.load_config_file()
        self._update_configs_from_dataset()
    
    def _load_config_from_file(self, config_type: str, file_path: Union[str, Path]) -> None:
        """Load configuration from a YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                if config_type == "project":
                    # OKS sigmas have one source of truth in dataset.yaml.
                    config.pop("kpt_oks_sigmas", None)
                if config_type == "other":
                    # Drop the obsolete YOLO task selector from both new and
                    # pre-existing project configs as they are loaded.
                    config.pop("task", None)
                if config_type == "other" and isinstance(config.get("training"), dict):
                    from animalposetracker.training.profiles import flatten_project_training_config

                    self._other_training_tree = deepcopy(config["training"])
                    config = flatten_project_training_config(
                        config,
                        model_type=self.project_config.get("model_type"),
                    )
                elif config_type == "other":
                    from animalposetracker.training.profiles import TRAINING_PROFILES

                    self._other_training_tree = deepcopy(TRAINING_PROFILES)
                setattr(self, f"{config_type}_config", config)
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load {config_type} config: {e}")
    
    def _load_config_file(self, config_type: str, config_path: Path) -> None:
        """Helper to load a configuration file."""
        self._load_config_from_file(config_type, config_path)

    def add_source_to_project(self, 
                              source_paths: Union[str, Path, List[Union[str, Path]]],
                              move_or_copy: str = 'copy') -> None:
        """Add source path(s) to the project after validation."""
        if not source_paths:
            raise ValueError("No source paths provided")
            
        sources = [source_paths] if isinstance(source_paths, (str, Path)) else source_paths
        if not isinstance(sources, list):
            raise ValueError("source_paths must be string, Path, or list thereof")
        if move_or_copy not in {"copy", "move"}:
            raise ValueError("move_or_copy must be either 'copy' or 'move'")

        current_sources = {
            str(Path(s).expanduser().resolve())
            for s in self.project_config.get("sources", [])
        }
        added_sources = []
        
        for source_path in sources:
            src_path = Path(source_path)
            if not src_path.exists():
                raise FileNotFoundError(f"path not found: {src_path}")

            destination = self._source_destination(src_path)
            stored_path = str(destination.resolve())
            if stored_path in current_sources:
                continue
            stored_path = str(
                self._handle_file_move_or_copy(
                    src_path, move_or_copy, destination=destination
                ).resolve()
            )
            added_sources.append(stored_path)
            current_sources.add(stored_path)
            
        if added_sources:
            self.project_config.setdefault("sources", []).extend(added_sources)
            self.update_config("project", {"sources": self.project_config["sources"]})
    
    def _source_destination(self, src_path: Union[str, Path]) -> Path:
        """Choose a stable per-source destination to avoid flattening/colliding inputs."""
        src_path = Path(src_path).expanduser().resolve()
        digest = hashlib.sha1(str(src_path).encode("utf-8")).hexdigest()[:8]
        if src_path.is_dir():
            return self._project_path / "sources" / "images" / f"{src_path.name}-{digest}"
        return (
            self._project_path / "sources" / "videos"
            / f"{src_path.stem}-{digest}{src_path.suffix}"
        )

    def _handle_file_move_or_copy(self, src_path, move_or_copy, destination=None):
        """
        Handle file or directory move or copy operations.

        Args:
            src_path (Path or str): The source file or directory path.
            move_or_copy (str): Operation type, 'copy' or 'move'.

        Returns:
            Path: The destination path if the operation is successful, None otherwise.
        """
        src_path = Path(src_path)

        try:
            if src_path.is_dir():
                dst_path = Path(destination) if destination is not None else self._source_destination(src_path)
                dst_path.mkdir(parents=True, exist_ok=True)
                for item in src_path.rglob('*'):
                    if item.is_file():
                        destination = dst_path / item.relative_to(src_path)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        self._perform_operation(item, destination, move_or_copy)
            else:
                dst_path = Path(destination) if destination is not None else self._source_destination(src_path)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    return dst_path
                self._perform_operation(src_path, dst_path, move_or_copy)
            return dst_path
        except (FileNotFoundError, PermissionError, FileExistsError):
            raise
        except Exception as e:
            raise RuntimeError(f"Could not {move_or_copy} source {src_path}: {e}") from e

    def _perform_operation(self, src, dst, operation):
        """
        Perform the copy or move operation.

        Args:
            src (Path): The source path.
            dst (Path): The destination path.
            operation (str): Operation type, 'copy' or 'move'.
        """
        if operation == 'copy':
            shutil.copy2(src, dst)
        elif operation =='move':
            shutil.move(src, dst)


    def save_configs(self, config_type: str = "all") -> None:
        """Save configurations to files."""
        configs_to_save = self.CONFIG_TYPES if config_type == "all" else {config_type}
        if config_type not in self.CONFIG_TYPES and config_type != "all":
            raise ValueError(f"Invalid config type. Must be one of {self.CONFIG_TYPES} or 'all'")
            
        for ct in configs_to_save:
            self._save_config(ct)

    def _save_config(self, config_type: str) -> None:
        """Save a single configuration to file."""
        config = getattr(self, f"{config_type}_config")
        if config_type == "other" and self._other_training_tree is not None:
            from animalposetracker.training.profiles import nested_other_config_from_flat

            config = nested_other_config_from_flat(
                config,
                self._other_training_tree,
                model_type=self.project_config.get("model_type", "AnimalRTPose"),
            )
        config_path = (
            self._project_path / f"{config_type}.yaml" 
            if config_type == "project" 
            else self._project_path / "configs" / f"{config_type}.yaml"
        )
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    config, 
                    f, 
                    indent=2, 
                    sort_keys=False, 
                    default_flow_style=False
                )
        except IOError as e:
            raise RuntimeError(f"Failed to save {config_type} config: {e}")

    def create_project_dirs(self) -> None:
        """Create the standard directory structure for the project."""
        for rel_dir in self.DEFAULT_DIRS:
            (self._project_path / rel_dir).mkdir(parents=True, exist_ok=True)
    
    def create_project_config(self) -> None:
        """Create and save the project configuration."""
        self._save_config("project")
    
    def create_dataset_config(self) -> None:
        """Create and save the dataset configuration."""
        self.dataset_config['path'] = str(self._project_path / "datasets")
        self._save_config("dataset")

    def create_model_config(self) -> None:
        """Create and save the model configuration."""
        model_type = self.project_config.get("model_type")
        if model_type not in MODEL_YAML_PATHS:
            raise ValueError(f"Invalid model type. Available: {list(MODEL_YAML_PATHS.keys())}")
        
        self._load_config_from_file("model", MODEL_YAML_PATHS[model_type])
        self._sync_model_config_from_dataset()
        self._save_config("model")

    def update_config(self, config_type: str, params: Dict[str, Any]) -> None:
        """Update configuration values."""
        if config_type not in self.CONFIG_TYPES:
            raise ValueError(f"Invalid config type. Must be one of {self.CONFIG_TYPES}")
            
        config = getattr(self, f"{config_type}_config")
        config.update({k: v for k, v in params.items() if k in config})
    
    def create_other_config(self) -> None:
        """Create any other missing configurations."""
        self._load_config_from_file("other", DEFAULT_CFG_PATH)

        from animalposetracker.training.profiles import (
            TRAINING_PROFILES,
            project_training_defaults,
        )

        model_type = self.project_config.get("model_type", "AnimalRTPose")
        self.other_config.update(project_training_defaults(model_type))
        self._other_training_tree = deepcopy(TRAINING_PROFILES)

        self.other_config.update({
            'model': str(self._project_path / "configs" / "model.yaml"),
            'data': str(self._project_path / "configs" / "dataset.yaml")
        })
        
        self._save_config("other")
    
    def _detect_pretrained(self):
        """
        Detects if a pre-trained model is available for the current project.
        If so, it updates the "pretrain" parameter in the "other" config.
        """
        pretrained = self.other_config.get("pretrained")
        if not pretrained:
            return
        
        pretrain_path = self._project_path / "pretrained"
        model_type = self.project_config.get("model_type")
        model_scale = self.project_config.get("model_scale")

        if model_type in ["AnimalRTPose", "AnimalViTPose", "AnimalRTPose-P6"]:
            weights_name = f"{model_type}-{model_scale}.pt"
        elif model_type in ["YOLOv8-Pose", "YOLO11-Pose"]:
            weights_name = f"{model_type}{model_scale}-pose.pt"
        elif model_type in ["YOLOv8-Pose-P6"]:
            weights_name = f"{model_type}{model_scale}-pose.pt"
        elif model_type in ["YOLOv12-Pose"]:
            weights_name = f"yolov12{model_scale}.pt"
        else:
            raise ValueError(f"Invalid model name {model_type}")
        
        weights_name = weights_name.lower()
        weights_path = pretrain_path / weights_name
        if model_type == "AnimalViTPose" and not weights_path.exists():
            # The training profile resolves the scale-specific MAE backbone checkpoint.
            # Do not turn pretrained off just because there is no bundled full-model file.
            return
        if not weights_path.exists():
            pretrained_urls = self.model_config.get("pretrained_urls") or {}
            if not pretrained_urls:
                # Existing projects may have copied their model yaml before checkpoint
                # URLs were colocated with the model definition. Fall back to the current
                # built-in model yaml without rewriting the user's project config.
                default_model_path = MODEL_YAML_PATHS.get(model_type)
                if default_model_path is not None and Path(default_model_path).is_file():
                    with Path(default_model_path).open("r", encoding="utf-8") as stream:
                        pretrained_urls = (yaml.safe_load(stream) or {}).get("pretrained_urls", {})
            url = pretrained_urls.get(str(model_scale).lower())
            if url is None:
                self.update_config("other", {"pretrained": False})
                self._save_config("other")
                return
            try:
                _download_google_drive_checkpoint(url, weights_path)
                print(f"Downloaded pre-trained weights to {weights_path}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download AnimalRTPose-{model_scale} pretrained weights from {url}: {exc}"
                ) from exc
            
        self.update_config("other", {"pretrained": str(weights_path)})
        self._save_config("other")

    def _update_and_save_config(self, mode, name, model=False, pretrained=False):
        """Update and save configuration."""
        if not model:
            new_model = str(self._project_path / "configs" / "model.yaml")
            self.update_config("other", {"pretrained": pretrained})
        else:
            new_model = self.project_path / "runs" / "train" / "weights" / "best.pt"
            self.update_config("other", {"pretrained": False})
        self.update_config("other", {"mode": mode})
        self.update_config("other", {"name": name})
        self.update_config("other", {"model": str(new_model)})
        self._save_config("other")

    def _execute_command(self, cmd):
        """Execute a command using subprocess."""
        self.process = subprocess.Popen(cmd, cwd=self._project_path)
        return self.process

    def train(self) -> None:
        """Train the model with AnimalPoseTracker's own PyTorch training worker."""
        from animalposetracker.workflows import unique_output_directory

        pretrained = self.other_config.get("pretrained")
        self._update_and_save_config("train", "train", pretrained=pretrained)
        self._detect_pretrained()
        run_root = Path(self.other_config.get("project") or "runs").expanduser()
        if not run_root.is_absolute():
            run_root = self._project_path / run_root
        run_path = unique_output_directory(
            run_root / str(self.other_config.get("name") or "train"),
            exist_ok=bool(self.other_config.get("exist_ok", False)),
        )
        self.other_config["name"] = run_path.name
        self._save_config("other")
        self.training_event_path = run_path / "training.jsonl"
        self._training_event_offset = (
            self.training_event_path.stat().st_size if self.training_event_path.is_file() else 0
        )
        stop_marker = self._project_path / ".animalposetracker_training_stop"
        if stop_marker.exists():
            stop_marker.unlink()
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.training.cli",
            "--config",
            "configs/other.yaml",
        ]
        self._execute_command(cmd)

    def request_training_stop(self) -> None:
        """Ask the custom training worker to save its state and stop safely."""
        if self.process is not None and self.process.poll() is None:
            (self._project_path / ".animalposetracker_training_stop").touch()

    def read_training_events(self) -> List[Dict[str, Any]]:
        """Read new JSON-lines status events from the active custom training worker."""
        if self.training_event_path is None or not self.training_event_path.is_file():
            return []
        events = []
        with self.training_event_path.open("rb") as stream:
            size = stream.seek(0, 2)
            if size < self._training_event_offset:
                self._training_event_offset = 0
            stream.seek(self._training_event_offset)
            for raw_line in stream:
                self._training_event_offset += len(raw_line)
                try:
                    events.append(json.loads(raw_line.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        return events

    def stop(self) -> None:
        """Stop the current process."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process = None

    def _resolve_project_weights(self, allow_onnx: bool = False) -> Path:
        """Use an explicitly selected checkpoint or the configured best checkpoint."""
        selected_model = Path(str(self.other_config.get("model") or "")).expanduser()
        if selected_model.suffix.lower() in {".pt", ".pth"}:
            return (selected_model if selected_model.is_absolute()
                    else self._project_path / selected_model).resolve()
        if allow_onnx and selected_model.suffix.lower() == ".onnx":
            return (selected_model if selected_model.is_absolute()
                    else self._project_path / selected_model).resolve()
        if selected_model.suffix.lower() not in {"", ".yaml", ".yml"}:
            raise ValueError(
                "Native project evaluation, prediction and export require a PyTorch .pt/.pth checkpoint"
            )

        run_root = Path(self.other_config.get("project") or "runs").expanduser()
        if not run_root.is_absolute():
            run_root = self._project_path / run_root
        run_name = str(self.other_config.get("name") or "train")
        return (run_root / run_name / "weights" / "best.pt").resolve()

    def _resolve_project_prediction_model(self) -> Union[Path, str]:
        """Resolve a native checkpoint or an exported model artifact for project prediction."""
        selected_value = str(self.other_config.get("model") or "").strip()
        if selected_value.startswith(("http://", "grpc://")):
            return selected_value
        selected_model = Path(selected_value).expanduser()
        model_path = (
            selected_model if selected_model.is_absolute()
            else self._project_path / selected_model
        ).resolve()
        if model_path.is_dir():
            return model_path
        if selected_model.suffix.lower() in {"", ".yaml", ".yml"}:
            return self._resolve_project_weights()
        if not model_path.exists():
            raise FileNotFoundError(f"Configured prediction model does not exist: {model_path}")
        return model_path

    def evaluate(self) -> None:
        """Evaluate a project checkpoint with AnimalPoseTracker's validator."""
        weights_path = self._resolve_project_weights()
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.training.cli",
            "--config",
            "configs/other.yaml",
            "--validate-only",
            "--weights",
            str(weights_path.resolve()),
            "--output-dir",
            "runs/val",
        ]
        self._execute_command(cmd)

    def predict(
        self,
        inference_source: Union[str, Path] = None,
        split: str = None,
        tracker: str = None,
        pose_filter: str = None,
    ) -> Path:
        """Predict a dataset/video and optionally track and filter its video poses."""
        from animalposetracker.workflows import unique_output_directory

        model_path = self._resolve_project_prediction_model()
        if inference_source is None:
            inference_source = self.other_config.get("source")
        output_root = Path(self.other_config.get("project") or "runs").expanduser()
        if not output_root.is_absolute():
            output_root = self._project_path / output_root
        self.prediction_output_dir = unique_output_directory(
            output_root / "predict",
            exist_ok=bool(self.other_config.get("exist_ok", False)),
        )
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.predict_cli",
            "--config",
            "configs/other.yaml",
            "--weights",
            str(model_path),
            "--output-dir",
            str(self.prediction_output_dir),
        ]
        if inference_source is not None:
            source_values = (
                list(inference_source)
                if isinstance(inference_source, (list, tuple))
                else [inference_source]
            )
            cmd.extend(["--source", *[str(value) for value in source_values]])
        prediction_split = split or self.other_config.get("predict_split")
        if prediction_split:
            cmd.extend(["--split", str(prediction_split)])
        if tracker:
            cmd.extend(["--tracker", str(tracker)])
        if pose_filter:
            cmd.extend(["--pose-filter", str(pose_filter)])
        self._execute_command(cmd)
        return self.prediction_output_dir

    def triangulate(
        self,
        predictions_by_camera: Dict[str, Any],
        cameras: Dict[str, Any],
        identity_map: Dict[str, Dict[Any, str]] = None,
        source_by_camera: Dict[str, str] = None,
        **options,
    ):
        """Triangulate synchronized tracked 2D prediction records from camera views."""
        from animalposetracker.postprocess import triangulate_multiview_predictions

        return triangulate_multiview_predictions(
            predictions_by_camera,
            cameras,
            identity_map=identity_map,
            source_by_camera=source_by_camera,
            **options,
        )

    def resume(self, resume_path: str) -> None:
        """Resume an AnimalPoseTracker training checkpoint."""
        if not resume_path:
            resume_path = str(self.project_path / "runs" / "train" / "weights" / "last.pt")
        checkpoint_path = Path(resume_path).expanduser().resolve()
        self._update_and_save_config("train", "train", pretrained=False)
        self.training_event_path = checkpoint_path.parent.parent / "training.jsonl"
        self._training_event_offset = (
            self.training_event_path.stat().st_size if self.training_event_path.is_file() else 0
        )
        stop_marker = self._project_path / ".animalposetracker_training_stop"
        if stop_marker.exists():
            stop_marker.unlink()
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.training.cli",
            "--config",
            "configs/other.yaml",
            "--resume",
            str(resume_path),
        ]
        self._execute_command(cmd)

    def export(self, format: Optional[str] = None) -> None:
        """Export a project checkpoint to an explicitly selected deployment format."""
        weights_path = self._resolve_project_weights()
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.export_cli",
            "--config",
            "configs/other.yaml",
            "--weights",
            str(weights_path),
            "--output-dir",
            "runs/export",
        ]
        if format is not None:
            cmd.extend(["--format", str(format)])
        self._execute_command(cmd)
