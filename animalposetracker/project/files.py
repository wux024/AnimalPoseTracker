"""Project projectfiles responsibilities."""

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

from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS, DEFAULT_CFG_PATH


class ProjectFilesMixin:
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

    def create_project_dirs(self) -> None:
        """Create the standard directory structure for the project."""
        for rel_dir in self.DEFAULT_DIRS:
            (self._project_path / rel_dir).mkdir(parents=True, exist_ok=True)
