"""Project projectjobs responsibilities."""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json
import os
import subprocess
import sys



class ProjectJobsMixin:
    def _update_and_save_config(self, name, model=False, pretrained=False):
        """Record what the training worker reads for this run.

        The name picks the run directory; a best.pt start means pruning the
        pretrained flag. (The old model/mode fields are gone: the model path
        is derived from the project structure now.)
        """
        if model:
            pretrained = False
        self.set_runtime_value("train.shared.runtime.pretrained", pretrained)
        self.set_runtime_value("common.name", name)
        self.save_configs("runtime")

    def _execute_command(self, cmd):
        """Execute a command using subprocess."""
        self.process = subprocess.Popen(cmd, cwd=self._project_path)
        return self.process

    def train(self) -> None:
        """Train the model with AnimalPoseTracker's own PyTorch training worker."""
        from animalposetracker.project.model_context import unique_output_directory

        pretrained = self.get_runtime_value("train.shared.runtime.pretrained")
        self._update_and_save_config("train", pretrained=pretrained)
        self._detect_pretrained()
        run_root = Path(self.get_runtime_value("common.project", "runs")).expanduser()
        if not run_root.is_absolute():
            run_root = self._project_path / run_root
        run_path = unique_output_directory(
            run_root / str(self.get_runtime_value("common.name", "train")),
            exist_ok=bool(self.get_runtime_value("common.exist_ok", False)),
        )
        self.set_runtime_value("common.name", run_path.name)
        self.save_configs("runtime")
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
            "configs/runtime.yaml",
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
                # A line without its terminator is a write in progress, not an event: the offset
                # stays put so the completed line is read on the next poll. Advancing past it
                # dropped that event for good, and the GUI polls once a second while the worker
                # is writing.
                if not raw_line.endswith(b"\n"):
                    break
                try:
                    event = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # A complete but unparseable line is skipped, not re-read forever.
                    self._training_event_offset += len(raw_line)
                    continue
                self._training_event_offset += len(raw_line)
                events.append(event)
        return events

    def stop(self) -> None:
        """Stop the current process."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process = None

    def evaluate(self) -> None:
        """Evaluate a project checkpoint with AnimalPoseTracker's validator."""
        from animalposetracker.project.model_context import unique_output_directory

        weights_path = self._resolve_project_weights()
        run_root = Path(self.get_runtime_value("common.project", "runs")).expanduser()
        if not run_root.is_absolute():
            run_root = self._project_path / run_root
        # The validator reports through the same structured events the trainer emits, so point at
        # the file it will write before launching it -- otherwise nobody can read the results.
        # Handing over a path that does not exist yet keeps the CLI's own uniquifying pass from
        # renaming it behind our back.
        run_path = unique_output_directory(
            run_root / "val", exist_ok=bool(self.get_runtime_value("common.exist_ok", False))
        )
        self.training_event_path = run_path / "training.jsonl"
        self._training_event_offset = 0
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.training.cli",
            "--config",
            "configs/runtime.yaml",
            "--validate-only",
            "--weights",
            str(weights_path.resolve()),
            "--output-dir",
            str(run_path),
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
            inference_source = self.get_runtime_value("predict.source")
        output_root = Path(self.get_runtime_value("common.project", "runs")).expanduser()
        if not output_root.is_absolute():
            output_root = self._project_path / output_root
        self.prediction_output_dir = unique_output_directory(
            output_root / "predict",
            exist_ok=bool(self.get_runtime_value("common.exist_ok", False)),
        )
        cmd = [
            sys.executable,
            "-m",
            "animalposetracker.prediction.cli",
            "--config",
            "configs/runtime.yaml",
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
        prediction_split = split or self.get_runtime_value("predict.split")
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
        self._update_and_save_config("train", pretrained=False)
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
            "configs/runtime.yaml",
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
            "configs/runtime.yaml",
            "--weights",
            str(weights_path),
            "--output-dir",
            "runs/export",
        ]
        if format is not None:
            cmd.extend(["--format", str(format)])
        self._execute_command(cmd)
