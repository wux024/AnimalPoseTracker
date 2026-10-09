"""Project projectjobs responsibilities."""

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


class ProjectJobsMixin:
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
        from animalposetracker.project.model_context import unique_output_directory

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
        from animalposetracker.project.model_context import unique_output_directory

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
            "animalposetracker.prediction.cli",
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
        from animalposetracker.postprocessing.workflows import triangulate_multiview_predictions

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
            "animalposetracker.export.cli",
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
