"""Project projectfiles responsibilities."""

from datetime import datetime
from pathlib import Path
from typing import List, Union
import hashlib
import shutil
import os



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
        # Keep the recorded path in step. The GUI sets `local_path` *after* pushing the form's
        # name/worker/model into the config, and without this the file kept the path computed
        # from the previous location -- a directory that never existed.
        config = getattr(self, "project_config", None)
        if isinstance(config, dict):
            config["project_path"] = str(self._project_path)

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
        # Refuse to write over an existing project: the directory name is
        # `name-worker-model-scale-date`, so a second create on the same day landed in the same
        # folder and silently replaced its configs.
        existing = self._project_path / "project.yaml"
        if existing.exists():
            raise FileExistsError(
                f"Project directory already exists; nothing was changed: {self._project_path}\n"
                "Use another project name, worker, or save location."
            )
        self._create_project_structure()
        if getattr(self, "pose3d", False):
            self._write_calibration_template()
        self.load_config_file()

    def enable_pose3d(self) -> None:
        """Turn an existing project into a 3D one: flag, calibration template, output folder.

        Projects are created without 3D by default and a capture plan can change afterwards.
        The flag lives in ``project.yaml``, so this rewrites that file and leaves every other
        key alone; the template stays empty, which the loader reads as "not calibrated yet"
        rather than as an identity rig.
        """
        if not (self._project_path / "project.yaml").is_file():
            raise RuntimeError("Load a project before enabling 3D reconstruction")
        if self.project_config.get("public_dataset"):
            raise ValueError(
                f"Project '{self._project_path.name}' was created from the public benchmark "
                f"'{self.project_config['public_dataset']}'; it has no multi-camera capture"
            )
        if self.project_config.get("pose3d"):
            return
        self.pose3d = True
        self.project_config["pose3d"] = True
        self.save_configs("project")
        self._write_calibration_template()
        self.create_project_dirs()

    def _write_calibration_template(self) -> None:
        """Write the empty camera calibration the 3D stage expects.

        The file is filled by hand (intrinsic/extrinsic computed with whichever tool the
        capture team uses); the loader validates it strictly, so an empty section is a
        clear "not filled in yet" rather than a silent identity calibration.
        """
        target = self._project_path / "configs" / "cameras.yaml"
        if target.exists():
            return
        target.write_text(
            "# Multi-view camera calibration for this project.\n"
            "#\n"
            "# Fill one block per camera; rotation and translation map WORLD coordinates\n"
            "# into the camera frame (x_cam = R @ x_world + t).\n"
            "#\n"
            "# world_unit: mm\n"
            "# cameras://n"
            "#   camA://n"
            "#     image_size: [1280, 1024]\n"
            "#     intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]\n"
            "#     distortion: {model: none}          # none | brown | fisheye\n"
            "#     rotation: [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]\n"
            "#     translation: [tx, ty, tz]\n"
            "world_unit: mm\n"
            "cameras: {}\n",
            encoding="utf-8",
        )

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
        directories = list(self.DEFAULT_DIRS)
        if getattr(self, "pose3d", False):
            directories.append("results/pose3d")
        for rel_dir in directories:
            (self._project_path / rel_dir).mkdir(parents=True, exist_ok=True)
