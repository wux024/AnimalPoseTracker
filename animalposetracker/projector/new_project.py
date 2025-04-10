import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS


class AnimalPoseTrackerProject:
    """Manage animal pose tracking projects 
    including configurations and directory structure."""
    
    # Constants for configuration types
    CONFIG_TYPES = {"project", "data", "model"}
    
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
            skeleton: Skeleton connections between keypoints (default: empty)
            oks_sigmas: OKS sigmas for keypoints (default: uniform distribution)
            classes_name: Names of classes (default: ['person'])
            sources: Source paths (default: None)
            date: Project creation date (default: current date)
        """
        # Initialize paths and basic attributes
        self.local_path = Path(local_path) if local_path else Path.cwd()
        self.date = date or datetime.now().strftime(r"%Y%m%d")
        self.project_name = project_name
        self.worker = worker
        self.model_type = model_type
        self.model_scale = model_scale
        
        # Keypoint        
        self.keypoints = keypoints
        self.visible = visible
        self.keypoints_name = keypoints_name or [f"kp_{i}" for i in range(keypoints)]
        self.skeleton = skeleton or []
        self.oks_sigmas = oks_sigmas or [1.0 / keypoints] * keypoints if keypoints > 0 else []
        
        # Class configuration
        self.classes = classes
        self.classes_name = classes_name or ['person']
        
        # Handle sources (single path or list of paths)
        self.sources = []
        if sources:
            if isinstance(sources, (str, Path)):
                self.sources = [Path(sources)]
            else:
                self.sources = [Path(s) if isinstance(s, str) else s for s in sources]

        # Generate project path and initialize configurations
        self.project_path = self._generate_project_path()
        self._initialize_configs()

    def _generate_project_path(self) -> Path:
        """Generate the project path based on naming convention."""
        return self.local_path / f"{self.project_name}-{self.worker}-{self.model_type}-{self.model_scale}-{self.date}"

    def _initialize_configs(self) -> None:
        """Initialize all configuration dictionaries."""
        # Common keypoint shape configuration
        kpt_shape = [self.keypoints, 3] if self.visible else [self.keypoints, 2]
        
        self.project_config = {
            "project_path": str(self.project_path),
            "project_name": self.project_name,
            "worker": self.worker,
            "model_type": self.model_type,
            "model_scale": self.model_scale,
            "date": self.date,
            "keypoints": self.keypoints,
            "visible": self.visible,
            "classes": self.classes,
            "keypoints_name": self.keypoints_name,
            "skeleton": self.skeleton,
            "oks_sigmas": self.oks_sigmas,
            "classes_name": self.classes_name,
            "sources": [str(s) for s in self.sources],
        }
        
        self.dataset_config = {
            'path': 'datasets',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'kpt_shape': kpt_shape,
            'flip_idx': [i for i in range(self.keypoints)],
            'names': dict(enumerate(self.classes_name)),
            'skeleton': self.skeleton,
            'oks_sigmas': self.oks_sigmas,
        }
        
        self.model_config = {
            'nc': self.classes,
            'kpt_shape': kpt_shape,
            'scales': None,
            'backbone': None,
            'head': None,
        }

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
        self.save_configs("all")

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
        
        self.update_config("project", self.dataset_config)
        self.update_config("dataset", {'path': str(self.project_path / "datasets")})
        self.update_config("model", self.dataset_config)
    
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
            "local_path", "project_name", "worker", "model_type", 
            "model_scale", "date", "keypoints", "visible", "classes",
            "keypoints_name", "skeleton", "oks_sigmas", "classes_name"
        ]
        
        self.local_path = Path(self.project_config["project_path"]).parent
        self.project_path = Path(self.project_config["project_path"])
        
        for attr in attrs[1:]:  # Skip local_path as we set it specially
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
        if any(s.resolve() == src_resolved for s in self.sources):
            print(f"Source already exists: {source_path}")
            return
            
        self.sources.append(source_path)
        self.update_config("project", {"sources": [str(s) for s in self.sources]})

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
        if self.model_type not in MODEL_YAML_PATHS:
            available = list(MODEL_YAML_PATHS.keys())
            raise ValueError(f"Invalid model type. Available: {available}")
        
        try:
            with open(MODEL_YAML_PATHS[self.model_type], "r") as f:
                self.model_config = yaml.safe_load(f) or {}
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load model config: {e}")
        
        # Update model-specific parameters
        self.model_config.update({
            'nc': self.classes,
            'kpt_shape': [self.keypoints, 3] if self.visible else [self.keypoints, 2]
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
            
        # self._save_config(config_type)


# if __name__ == '__main__':
#     project = AnimalPoseTrackerProject(
#         local_path='D:/wux024/AnimalPoseTrackerProject',
#         project_name='person',
#         worker='Adam',
#         model_type='AnimalRTPose',
#         model_scale='N',
#         keypoints=17,
#         visible=True,
#         classes=1,
#         keypoints_name=None,
#         skeleton=None,
#         oks_sigmas=None,
#         classes_name=None,
#         sources=None,
#         date=None
#     )
#     project.print_project_info()
#     project.create_new_project()
#     project.print_project_info()

#     project.load_project_config('D:\wux024\AnimalPoseTrackerProject\AP10K-Adam-AnimalRTPose-N-20250410\configs\project.yaml')
#     project.print_project_info()

#     project.create_public_dataset_project('APT36K')

#     project.print_project_info()