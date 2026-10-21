"""Project projectconfiguration responsibilities."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import os
import yaml

from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS


class ProjectConfigurationMixin:
    def _validate_keypoints_params(
        self,
        keypoints: int,
        keypoint_names: Optional[List[str]],
        skeleton: Optional[List[List[int]]]
    ) -> None:
        """Validate keypoints-related parameters."""
        if keypoint_names is not None and len(keypoint_names) != keypoints:
            raise ValueError(
                f"keypoint_names length ({len(keypoint_names)}) must match keypoints ({keypoints})"
            )

        if skeleton is not None:
            for connection in skeleton:
                if any(idx >= keypoints for idx in connection):
                    raise ValueError(
                        f"Skeleton contains invalid keypoint index (max {keypoints-1})"
                    )

    def _init_config(self, default: Dict[str, Any]) -> Dict[str, Any]:
        return default.copy()

    def load_config_file(self) -> None:
        """Load the project configuration from file."""
        config_files = {
            "project": self._project_path / "project.yaml",
            "dataset": self._project_path / "configs" / "dataset.yaml",
            "model": self._project_path / "configs" / "model.yaml",
            "runtime": self._project_path / "configs" / "runtime.yaml",
        }

        for config_type, config_file in config_files.items():
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            self._load_config_file(config_type, config_file)
        # Once the project exists the file is the truth: `pose3d` decides whether the project
        # carries results/pose3d and whether the 3D page is usable, so a loaded rig must not be
        # masked by the constructor's default.
        self.pose3d = bool((self.project_config or {}).get("pose3d", False))

    def create_public_dataset_project(self, dataname: str = 'AP10K') -> None:
        """Create a project using a public dataset configuration."""
        if dataname not in DATA_YAML_PATHS:
            raise ValueError(f"Invalid dataset name. Available: {list(DATA_YAML_PATHS.keys())}")

        # Recorded before the configs are written: this field is what keeps the 3D stage out
        # of a benchmark project later on, and the CLI's `create` records it the same way.
        self.project_config["public_dataset"] = dataname
        self._create_project_structure()
        self._load_config_from_file("dataset", DATA_YAML_PATHS[dataname])
        self._update_configs_from_dataset()
        self.load_config_file()

    def _update_configs_from_dataset(self) -> None:
        """Update configurations based on dataset settings.

        The dataset file is the shape's single source now: the project file
        keeps identity only, and the model file takes its output shape from
        the dataset here.
        """
        self.update_config("project", {"project_path": str(self._project_path)})
        self.update_config("dataset", {'path': str(self._project_path / "datasets")})
        self._sync_model_config_from_dataset()
        self.save_configs()

    def _sync_model_config_from_dataset(self) -> None:
        """Keep the model YAML's output shape aligned with the project dataset."""
        model_nc = (
            1 if self.project_config.get("model_type") == "AnimalViTPose"
            else len(self.dataset_config.get("names") or {}) or 1
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
                setattr(self, f"{config_type}_config", config)
        except (IOError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load {config_type} config: {e}")

    def _load_config_file(self, config_type: str, config_path: Path) -> None:
        """Helper to load a configuration file."""
        self._load_config_from_file(config_type, config_path)

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

        if config_type == "project" and any(
            key in params for key in ("project_name", "worker", "model_type", "model_scale", "date")
        ):
            # Those five components build the directory name, so the recorded path has to be
            # recomputed here. It used to keep whatever it was at construction time, and
            # `project.yaml` then pointed at a directory that never existed.
            self._project_path = self._generate_project_path()
            config["project_path"] = str(self._project_path)

    def create_runtime_config(self) -> None:
        """Create the runtime configuration from the package defaults.

        A snapshot, not a diff: the project file carries every registered
        default so it reads (and edits) without consulting the package, and a
        later package update never rewrites a project's behaviour silently.
        """
        from animalposetracker.config import default_subtree

        self.runtime_config = {
            "common": default_subtree("common"),
            "train": default_subtree("train"),
            "predict": default_subtree("predict"),
            "export": default_subtree("export"),
        }
        self._save_config("runtime")

    def set_runtime_value(self, dotted_path: str, value: Any) -> None:
        """Write one value into the runtime tree by dotted path."""
        parts = [part for part in str(dotted_path).split(".") if part]
        if len(parts) < 2 or parts[0] not in self.runtime_config:
            raise ValueError(f"Unknown runtime path: {dotted_path}")
        node = self.runtime_config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def get_runtime_value(self, dotted_path: str, default: Any = None) -> Any:
        """Read one value from the runtime tree by dotted path."""
        node = self.runtime_config
        for part in str(dotted_path).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def apply_runtime_overrides(self, values: Dict[str, Any]) -> None:
        """Place flat create-time values into the runtime tree.

        Command-line spellings are canonical now; each key lands in the
        section the trainer's flatten actually reads it from.
        """
        from animalposetracker.training.profiles import train_key_targets

        model_type = self.project_config.get("model_type", "AnimalRTPose")
        targets = train_key_targets(model_type)
        for key, value in (values or {}).items():
            if key in ("device", "project", "name", "exist_ok"):
                self.set_runtime_value(f"common.{key}", value)
            elif key in ("pretrained", "pretrained_weights"):
                self.set_runtime_value(f"train.shared.runtime.{key}", value)
            elif key in targets:
                self.set_runtime_value(targets[key], value)
