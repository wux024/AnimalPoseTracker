import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS, DEFAULT_CFG_PATH

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
        "datasets/labels",
        "datasets/annotations",
        "pretrained",
        "runs",
        "sources",
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
                 oks_sigmas: Optional[List[float]] = None, 
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
            oks_sigmas: OKS sigmas for keypoints (default: uniform distribution)
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
        
        if oks_sigmas is None:
            oks_sigmas = [1.0 / keypoints] * keypoints  # Default uniform distribution

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
            "oks_sigmas": oks_sigmas,
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
            'oks_sigmas': oks_sigmas,
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
        
        # Generate project path and initialize configurations
        self.local_path = Path(local_path) if local_path else Path.cwd()
        self.project_config['project_path'] = str(self.project_path)
    
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
    
    def _init_config(self, config_type: str, default: Dict[str, Any]) -> Dict[str, Any]:
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
            config['date']
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
            "project": self.project_path / "project.yaml",
            "dataset": self.project_path / "configs" / "dataset.yaml",
            "model": self.project_path / "configs" / "model.yaml",
            "other": self.project_path / "configs" / "other.yaml",
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
            "visible": self.dataset_config['kpt_shape'][1] == 3
        })
        
        self._update_configs_from_dataset()
        self.load_config_file()
    
    def _update_configs_from_dataset(self) -> None:
        """Update configurations based on dataset settings."""
        self.update_config("project", self.dataset_config)
        self.update_config("dataset", {'path': str(self.project_path / "datasets")})
        self.update_config("model", self.dataset_config)
    
    def load_project_config(self, config_path: Union[str, Path]) -> None:
        """Load a project configuration from file."""
        self._load_config_from_file("project", config_path)
        self.local_path = Path(config_path).parent.parent
        self.load_config_file()
    
    def _load_config_from_file(self, config_type: str, file_path: Union[str, Path]) -> None:
        """Load configuration from a YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                setattr(self, f"{config_type}_config", config)
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load {config_type} config: {e}")
    
    def _load_config_file(self, config_type: str, config_path: Path) -> None:
        """Helper to load a configuration file."""
        self._load_config_from_file(config_type, config_path)

    def add_source_to_project(self, source_paths: Union[str, Path, List[Union[str, Path]]]) -> None:
        """Add source path(s) to the project after validation."""
        if not source_paths:
            raise ValueError("No source paths provided")
            
        sources = [source_paths] if isinstance(source_paths, (str, Path)) else source_paths
        if not isinstance(sources, list):
            raise ValueError("source_paths must be string, Path, or list thereof")

        current_sources = [Path(s).resolve() for s in self.project_config.get("sources", [])]
        added_sources = []
        
        for source_path in sources:
            src_path = Path(source_path)
            if not src_path.exists():
                raise FileNotFoundError(f"path not found: {src_path}")
                
            src_resolved = src_path.resolve()
            if any(s == src_resolved for s in current_sources):
                print(f"Source already exists: {src_path}")
                continue
                
            added_sources.append(str(src_path))
            current_sources.append(src_resolved)
            
        if added_sources:
            self.project_config.setdefault("sources", []).extend(added_sources)
            self.update_config("project", {"sources": self.project_config["sources"]})

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
        config_path = (
            self.project_path / f"{config_type}.yaml" 
            if config_type == "project" 
            else self.project_path / "configs" / f"{config_type}.yaml"
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
            (self.project_path / rel_dir).mkdir(parents=True, exist_ok=True)
    
    def create_dataset_config(self) -> None:
        """Create and save the dataset configuration."""
        self.dataset_config['path'] = str(self.project_path / "datasets")
        self._save_config("dataset")

    def create_model_config(self) -> None:
        """Create and save the model configuration."""
        model_type = self.project_config.get("model_type")
        if model_type not in MODEL_YAML_PATHS:
            raise ValueError(f"Invalid model type. Available: {list(MODEL_YAML_PATHS.keys())}")
        
        self._load_config_from_file("model", MODEL_YAML_PATHS[model_type])
        self.model_config.update({
            'nc': self.project_config["classes"],
            'kpt_shape': self.dataset_config['kpt_shape']
        })
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
        
        model_scale = self.project_config['model_scale'].lower()
        self.other_config.update({
            'model': str(self.project_path / "configs" / f"model-{model_scale}.yaml"),
            'data': str(self.project_path / "configs" / "dataset.yaml")
        })
        
        self._save_config("other")