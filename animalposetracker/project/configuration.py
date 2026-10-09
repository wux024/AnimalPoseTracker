"""Project projectconfiguration responsibilities."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import hashlib
import json
import shutil
import os
import tempfile
import subprocess
import sys
import yaml

from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS, DEFAULT_CFG_PATH


class ProjectConfigurationMixin:
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
