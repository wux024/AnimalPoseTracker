"""Pretrained checkpoint acquisition and project weight resolution."""

import os
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Union
import yaml

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


class ProjectWeightsMixin:
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

        if model_type in ["AnimalRTPose", "AnimalViTPose"]:
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
                self._download_pretrained_checkpoint(url, weights_path)
                print(f"Downloaded pre-trained weights to {weights_path}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download AnimalRTPose-{model_scale} pretrained weights from {url}: {exc}"
                ) from exc

        self.update_config("other", {"pretrained": str(weights_path)})
        self._save_config("other")

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
