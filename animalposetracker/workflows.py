"""Shared model-loading helpers for native inference and export workflows."""

from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import yaml

from animalposetracker.training.checkpoint import load_model_weights
from animalposetracker.training.config import TrainingConfig
from animalposetracker.training.profiles import (
    configure_animalvitpose_model,
    flatten_project_training_config,
)


def read_yaml_mapping(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_device(torch, requested: Any):
    """Resolve one inference device; a device list uses its first entry."""
    if isinstance(requested, (list, tuple)):
        requested = requested[0] if requested else "auto"
    value = str(requested or "auto").strip().lower()
    if "," in value:
        value = value.split(",", 1)[0].strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot use CUDA")
    return device


def unique_output_directory(path: Path, exist_ok: bool = False) -> Path:
    """Avoid replacing prior workflow results unless the project allows it."""
    path = Path(path).expanduser()
    if exist_ok or not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            return candidate
        index += 1


def load_project_context(config_path: Path) -> Tuple[TrainingConfig, Dict[str, Any]]:
    """Read a project's training, data and model context without loading PyTorch weights."""
    config_path = Path(config_path).expanduser().resolve()
    project_dir = config_path.parent.parent
    project_path = project_dir / "project.yaml"
    project_values = read_yaml_mapping(project_path) if project_path.is_file() else {}
    other_values = flatten_project_training_config(
        read_yaml_mapping(config_path),
        model_type=project_values.get("model_type"),
    )
    config = TrainingConfig.from_mapping(other_values, project_dir=project_dir)
    if config.model is None or not config.model.is_file():
        raise FileNotFoundError(f"Model configuration does not exist: {config.model}")
    if config.data is None or not config.data.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {config.data}")
    context = {
        "project_dir": project_dir,
        "project": project_values,
        "other": other_values,
        "dataset": read_yaml_mapping(config.data),
        "model_spec": read_yaml_mapping(config.model),
        "weights": None,
        "load_report": None,
    }
    return config, context


def configure_project_model_spec(config: TrainingConfig, context: Dict[str, Any]):
    """Materialize model architecture values from the project, model and dataset configs."""
    from animalposetracker.training.cli import _head_config, _set_simcc_keypoint_count

    model_spec = deepcopy(context["model_spec"])
    dataset_spec = context["dataset"]
    dataset_shape = dataset_spec.get("kpt_shape")
    if isinstance(dataset_shape, (list, tuple)) and len(dataset_shape) == 2:
        model_spec["kpt_shape"] = list(dataset_shape)
    class_names = dataset_spec.get("names") or {}
    class_count = len(class_names) if isinstance(class_names, (dict, list, tuple)) else int(
        model_spec.get("nc", 1)
    )

    scale = str(context["project"].get("model_scale") or "n").lower()
    head_name, _head_entry = _head_config(model_spec)
    if head_name == "SimCCHead":
        configure_animalvitpose_model(model_spec, scale, config.image_size)
        if not isinstance(dataset_shape, (list, tuple)) or len(dataset_shape) != 2:
            raise ValueError("AnimalViTPose dataset.yaml must define kpt_shape=[count, dimensions]")
        _set_simcc_keypoint_count(model_spec, int(dataset_shape[0]), dataset_shape)
        model_spec["nc"] = 1
    elif head_name == "YOLOPoseHead":
        model_spec["nc"] = 1 if config.single_cls else class_count
    else:
        raise NotImplementedError(f"Native inference does not support model head {head_name!r}")
    return model_spec, head_name, scale


def load_project_model(
    config_path: Path,
    weights_path: Path,
    device: Optional[str] = None,
) -> Tuple[Any, TrainingConfig, Dict[str, Any]]:
    """Build a project model, load its weights, and return model/config/context."""
    import torch

    from animalposetracker.nn import build_model
    config, context = load_project_context(config_path)
    model_spec, head_name, scale = configure_project_model_spec(config, context)

    model = build_model(model_spec, scale=scale)
    target_device = resolve_device(torch, device or config.device)
    model.to(target_device)
    model.eval()
    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
    load_report = load_model_weights(
        weights_path,
        model,
        map_location=target_device,
        strict=False,
    )
    context.update({
        "model_spec": model_spec,
        "head_name": head_name,
        "scale": scale,
        "device": target_device,
        "weights": weights_path,
        "load_report": load_report,
    })
    return model, config, context


__all__ = [
    "load_project_model",
    "load_project_context",
    "configure_project_model_spec",
    "read_yaml_mapping",
    "resolve_device",
    "unique_output_directory",
]
