import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from warnings import warn
from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS


class AnimalPoseTrackerProject:
    """Manage animal pose tracking projects including configurations and directory structure."""
    
    # Constants for configuration types
    CONFIG_TYPES = {"project", "dataset", "model"}
    
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
            "date:": date if date is not None else datetime.now().strftime(r"%Y%m%d"),
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
        
        # Generate project path and initialize configurations
        self._local_path = Path(local_path) if local_path else Path.cwd()
        self._project_path = self._generate_project_path()
        self.project_config['project_path'] = self._generate_project_path()

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
        project_name = self.project_config.get("project_name", "person")
        worker = self.project_config.get("worker", "Adam")
        model_type = self.project_config.get("model_type", "AnimalRTPose")
        model_scale = self.project_config.get("model_scale", "N")
        date = self.project_config.get("date", datetime.now().strftime(r"%Y%m%d"))
        return self.local_path / f"{project_name}-{worker}-{model_type}-{model_scale}-{date}"

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
        self.create_project_dirs()
        self.create_project_config()
        self.create_dataset_config()
        self.create_model_config()

    def create_public_dataset_project(self, dataname: str = 'AP10K') -> None:
        """
        Create a project using a public dataset configuration.
        
        Args:
            dataname: Name of public dataset (default: 'AP10K')
            
        Raises:
            ValueError: If dataset name is invalid
            RuntimeError: If config loading fails
        """
        if dataname not in DATA_YAML_PATHS:
            available = list(DATA_YAML_PATHS.keys())
            raise ValueError(f"Invalid dataset name. Available: {available}")
        
        self.create_project_dirs()
        
        try:
            with open(DATA_YAML_PATHS[dataname], "r") as f:
                self.dataset_config = yaml.safe_load(f) or {}
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load dataset config: {e}")
        
        self.project_config["classes"] = len(self.dataset_config['names'].keys())
        self.project_config["classes_name"] = list(self.dataset_config['names'].values())
        self.project_config["keypoints"] = self.dataset_config['kpt_shape'][0]
        self.project_config["visible"] = self.dataset_config['kpt_shape'][1] == 3
        
        self.update_config("project", self.dataset_config)
        self.update_config("dataset", {'path': str(self.project_path / "datasets")})
        self.update_config("model", self.dataset_config)
        self.create_project_config()
        self.create_dataset_config()
        self.create_model_config()
    
    def load_project_config(self, config_path: Union[str, Path]) -> None:
        """
        Load a project configuration from file.
        
        Args:
            config_path: Path to project configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            RuntimeError: If config loading fails
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, "r") as f:
                self.project_config = yaml.safe_load(f) or {}
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load project config: {e}")
        
        # Update instance variables from loaded config
        self._update_from_config()
        
        # Load other configs if they exist
        for config_type in ["dataset", "model"]:
            config_file = self.project_path / "configs" / f"{config_type}.yaml"
            if config_file.exists():
                self._load_config_file(config_type, config_file)
            else:
                getattr(self, f"create_{config_type}_config")()

    def _update_from_config(self) -> None:
        """Update instance variables from loaded project config."""
        attrs = [
            "project_name", "worker", "model_type", 
            "model_scale", "date", "keypoints", "visible", "classes",
            "keypoints_name", "skeleton", "oks_sigmas", "classes_name"
        ]
        
        self._local_path = Path(self.project_config["project_path"]).parent
        self._project_path = Path(self.project_config["project_path"])
        
        for attr in attrs:
            if attr in self.project_config:
                setattr(self, attr, self.project_config[attr])

    def _load_config_file(self, config_type: str, config_path: Path) -> None:
        """Helper to load a configuration file."""
        try:
            with open(config_path, "r") as f:
                setattr(self, f"{config_type}_config", yaml.safe_load(f) or {})
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load {config_type} config: {e}")

    def add_source_to_project(self, source_path: Union[str, Path]) -> None:
        """
        Add a source path to the project.
        
        Args:
            source_path: Path to source to add
            
        Raises:
            FileNotFoundError: If source path doesn't exist
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
            
        # Check for duplicates using path resolution
        src_resolved = source_path.resolve()
        current_sources = [Path(s).resolve() for s in self.project_config["sources"]]
        
        if any(s == src_resolved for s in current_sources):
            print(f"Source already exists: {source_path}")
            return
            
        self.project_config["sources"].append(str(source_path))
        self.update_config("project", {"sources": self.project_config["sources"]})

    def save_configs(self, config_type: str = "all") -> None:
        """
        Save configurations to files.
        
        Args:
            config_type: Which configs to save ('project', 'dataset', 'model', or 'all')
            
        Raises:
            ValueError: If invalid config type is provided
        """
        if config_type == "all":
            configs_to_save = self.CONFIG_TYPES
        elif config_type in self.CONFIG_TYPES:
            configs_to_save = {config_type}
        else:
            raise ValueError(f"Invalid config type. Must be one of {self.CONFIG_TYPES} or 'all'")
            
        for ct in configs_to_save:
            self._save_config(ct)

    def _save_config(self, config_type: str) -> None:
        """Helper method to save a single configuration."""
        config = getattr(self, f"{config_type}_config")
        config_path = self.project_path / "configs" / f"{config_type}.yaml"
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
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
            dir_path = self.project_path / rel_dir
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_project_config(self) -> None:
        """Create and save the project configuration."""
        self._save_config("project")

    def create_dataset_config(self) -> None:
        """Create and save the dataset configuration."""
        self.dataset_config['path'] = str(self.project_path / "datasets")
        self._save_config("dataset")

    def create_model_config(self) -> None:
        """Create and save the model configuration.
        
        Raises:
            ValueError: If model type is invalid
            RuntimeError: If config loading fails
        """
        model_type = self.project_config.get("model_type")
        if model_type not in MODEL_YAML_PATHS:
            available = list(MODEL_YAML_PATHS.keys())
            raise ValueError(f"Invalid model type. Available: {available}")
        
        try:
            with open(MODEL_YAML_PATHS[model_type], "r") as f:
                self.model_config = yaml.safe_load(f) or {}
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load model config: {e}")
        
        # Update model-specific parameters
        self.model_config.update({
            'nc': self.project_config["classes"],
            'kpt_shape': [self.project_config["keypoints"], 3] 
                if self.project_config["visible"] 
                else [self.project_config["keypoints"], 2]
        })
        
        self._save_config("model")

    def update_config(self, config_type: str, params: Dict[str, Any]) -> None:
        """
        Update configuration values.
        
        Args:
            config_type: Type of config ('project', 'dataset', or 'model')
            params: Dictionary of key-value pairs to update
            
        Raises:
            ValueError: If invalid config type is provided
        """
        if config_type not in self.CONFIG_TYPES:
            raise ValueError(f"Invalid config type. Must be one of {self.CONFIG_TYPES}")
            
        config = getattr(self, f"{config_type}_config")
        valid_keys = set(config.keys())
        
        for key, value in params.items():
            if key not in valid_keys:
                continue
            config[key] = value