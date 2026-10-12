"""Command-line project creation and configuration."""

import argparse
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import yaml

from animalposetracker.cfg import DATA_YAML_PATHS
from .api import AnimalPoseTrackerProject


def _make_create_parser() -> argparse.ArgumentParser:
    from animalposetracker.training.cli import (
        _CLI_ALIASES,
        _CLI_COMMAND_FIELDS,
        _CLI_FIELD_TO_KEY,
        _CLI_INTERNAL_FIELDS,
        _CLI_SPECIAL_TYPES,
        _add_bool_options,
        _parse_cache_value,
        _parse_pretrained_value,
    )
    from animalposetracker.training.config import TrainingConfig

    parser = argparse.ArgumentParser(
        prog="animalpose-cli create",
        description="Create a project from a dataset YAML without copying the dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Built-in dataset name (for example trimouse) or a dataset YAML file",
    )
    parser.add_argument(
        "--dataset-root",
        help="Dataset directory (defaults to <project>/datasets)",
    )
    parser.add_argument("--workspace", required=True, help="Directory where the project is created")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument("--worker", default="user")
    parser.add_argument("--model", default="AnimalRTPose")
    parser.add_argument("--scale", default="N")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--annotation-format", choices=("auto", "coco", "yolo"), default="auto")

    # Match train's config flags and parsing rules, while retaining create's
    # own --model/--scale names for model type and model scale.
    excluded_fields = _CLI_INTERNAL_FIELDS | {
        "model", "data", "pretrained_weights", "project_dir", "output_dir",
        "resume_from", "train_dataset_size", "resolved_optimizer",
        "resolved_learning_rate", "resolved_momentum", "resolved_weight_decay",
    }
    field_types = {
        field.name: _CLI_SPECIAL_TYPES.get(field.name, type(field.default))
        for field in dataclass_fields(TrainingConfig)
    }
    for name in _CLI_COMMAND_FIELDS:
        if name in excluded_fields or name == "cache":
            continue
        dest = "augmentation_scale" if name == "scale" else name
        if name == "scale":
            flags = ("--augment-scale",)
        else:
            flags = tuple(dict.fromkeys((f"--{name.replace('_', '-')}", *_CLI_ALIASES.get(name, ()))))
        value_type = field_types[name]
        if value_type is bool:
            _add_bool_options(parser, flags, dest)
        elif name == "scale":
            parser.add_argument(
                *flags, dest=dest, type=value_type, default=argparse.SUPPRESS,
                help="Training scale augmentation; --scale selects the model scale.",
            )
        else:
            parser.add_argument(*flags, dest=dest, type=value_type, default=argparse.SUPPRESS)

    parser.add_argument("--cache", nargs="?", const=True, type=_parse_cache_value,
                        default=argparse.SUPPRESS)
    parser.add_argument("--no-cache", dest="cache", action="store_const", const=False,
                        default=argparse.SUPPRESS)
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument(
        "--pretrained",
        nargs="?",
        const=True,
        type=_parse_pretrained_value,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=(
            "Start from pretrained weights. Optionally record a local "
            "checkpoint path in configs/other.yaml; nothing is checked or "
            "downloaded at create time."
        ),
    )
    pretrained.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_const",
        const=False,
        default=argparse.SUPPRESS,
        help="Train from scratch.",
    )
    pretrained.add_argument(
        "--pretrained-weights", dest="pretrained_weights", default=argparse.SUPPRESS,
        help="Record a local checkpoint path in configs/other.yaml.",
    )
    parser.add_argument("--no-gradient-clip", dest="gradient_clip_norm",
                        action="store_const", const=None, default=argparse.SUPPRESS)
    parser.add_argument("--quiet", dest="verbose", action="store_const", const=False,
                        default=argparse.SUPPRESS)
    return parser


def _collect_create_training_overrides(args) -> dict:
    from animalposetracker.training.cli import (
        _CLI_COMMAND_FIELDS,
        _CLI_FIELD_TO_KEY,
        _CLI_INTERNAL_FIELDS,
    )
    from animalposetracker.training.profiles import _PROJECT_KEY_ALIASES

    excluded_fields = _CLI_INTERNAL_FIELDS | {
        "model", "data", "pretrained_weights", "project_dir", "output_dir",
        "resume_from", "train_dataset_size", "resolved_optimizer",
        "resolved_learning_rate", "resolved_momentum", "resolved_weight_decay",
    }
    values = {}
    for name in _CLI_COMMAND_FIELDS:
        if name in excluded_fields:
            continue
        dest = "augmentation_scale" if name == "scale" else name
        if hasattr(args, dest):
            key = _CLI_FIELD_TO_KEY[name]
            values[_PROJECT_KEY_ALIASES.get(key, key)] = getattr(args, dest)
    if hasattr(args, "cache"):
        values["cache"] = args.cache
    if hasattr(args, "pretrained"):
        values["pretrained"] = args.pretrained
        if args.pretrained is False:
            values["pretrained_weights"] = None
        elif isinstance(args.pretrained, str):
            values["pretrained_weights"] = args.pretrained
    if hasattr(args, "pretrained_weights"):
        values["pretrained_weights"] = args.pretrained_weights
    return values


def _normalize_dataset_name(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _resolve_dataset_path(value: str) -> Path:
    """Resolve a built-in dataset alias or an explicit YAML path."""
    normalized_value = _normalize_dataset_name(value)
    for name, path in DATA_YAML_PATHS.items():
        if normalized_value in {
            _normalize_dataset_name(name),
            _normalize_dataset_name(path.stem),
        }:
            return path
    return Path(value).expanduser().resolve()


def _project_path_from_args(args: argparse.Namespace) -> Path:
    """Build the project path using the same naming rule as AnimalPoseTrackerProject."""
    project_dir_name = "-".join((
        args.name,
        args.worker,
        args.model,
        args.scale,
        str(args.date),
    ))
    return Path(args.workspace).expanduser().resolve() / project_dir_name


def _load_dataset_config(path: Path, dataset_root: Path, annotation_format: str) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")

    config["path"] = str(dataset_root)
    if annotation_format != "auto":
        config["annotation_format"] = annotation_format
    elif not config.get("annotation_format"):
        annotation_dir = dataset_root / "annotations"
        config["annotation_format"] = "coco" if any(annotation_dir.glob("*.json")) else "yolo"
    return config


def _create_project(args: argparse.Namespace) -> AnimalPoseTrackerProject:
    dataset_path = _resolve_dataset_path(args.dataset)
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else _project_path_from_args(args) / "datasets"
    )
    dataset_config = _load_dataset_config(dataset_path, dataset_root, args.annotation_format)

    keypoint_count, keypoint_dims = dataset_config["kpt_shape"]
    class_names = {int(key): str(value) for key, value in dataset_config["names"].items()}
    ordered_names = [class_names[index] for index in sorted(class_names)]
    keypoint_names = dataset_config.get(
        "keypoint_names", [f"kpt_{index}" for index in range(keypoint_count)]
    )

    project = AnimalPoseTrackerProject(
        local_path=Path(args.workspace).expanduser().resolve(),
        project_name=args.name,
        worker=args.worker,
        model_type=args.model,
        model_scale=args.scale,
        keypoints=keypoint_count,
        visible=(keypoint_dims == 3),
        classes=len(class_names),
        keypoints_name=keypoint_names,
        skeleton=dataset_config.get("skeleton", []),
        kpt_oks_sigmas=dataset_config.get("kpt_oks_sigmas"),
        classes_name=ordered_names,
        date=args.date,
    )
    project.create_new_project()
    # dataset.yaml now derives every shape key from the arguments the project was built with, so
    # only carry over what the dataset file adds on top of them (annotation_format, splits, root).
    project.dataset_config.update({
        key: value
        for key, value in dataset_config.items()
        if key not in {"kpt_shape", "flip_idx", "names", "skeleton", "kpt_oks_sigmas"}
    })
    project.dataset_config["path"] = str(dataset_root)
    project.other_config.update({
        "val": True,
        "plots": True,
        "project": "runs",
        "name": "train",
        "exist_ok": False,
    })
    project.other_config.update(_collect_create_training_overrides(args))
    # Pretrained weight resolution is deferred to training time so project
    # creation never performs file checks or network downloads.
    project.save_configs("all")
    _print_create_banner(args, project, dataset_root)
    return project


def _print_create_banner(args, project: AnimalPoseTrackerProject, dataset_root: Path) -> None:
    from animalposetracker.console import (
        describe_pretrained,
        environment_line,
        format_summary_block,
    )

    class_names = project.project_config.get("classes_name") or []
    keypoints = project.project_config.get("keypoints")
    training = project.other_config
    pretrained = training.get("pretrained", False)
    if training.get("pretrained_weights") and pretrained is not False:
        pretrained = training["pretrained_weights"]
    print()
    print(environment_line())
    print()
    print(format_summary_block([
        ("Project", str(project.project_path)),
        ("Model", f"{args.model}-{args.scale}"),
        ("Dataset", str(dataset_root)),
        ("Annotations", str(project.dataset_config["annotation_format"])),
        ("Keypoints", str(keypoints)),
        ("Classes", f"{len(class_names)} ({', '.join(map(str, class_names))})"),
        (
            "Training",
            f"epochs={training.get('epochs', '?')} "
            f"batch={training.get('batch', training.get('batch_size', '?'))} "
            f"imgsz={training.get('imgsz', training.get('image_size', '?'))} "
            f"workers={training.get('workers', training.get('num_workers', '?'))} "
            f"device={training.get('device', '?')}",
        ),
        ("Pretrained", describe_pretrained(pretrained)),
    ]))
    print()
    print(f"Next: animalpose-cli train --config {project.project_path / 'configs' / 'other.yaml'}")


def _make_load_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli load",
        description="Validate and display an existing AnimalPoseTracker project without opening the GUI.",
    )
    parser.add_argument(
        "project_path",
        help="Project directory, project.yaml, or configs/other.yaml",
    )
    return parser


def run_load(argv=None) -> int:
    parser = _make_load_parser()
    args = parser.parse_args(argv)
    selected_path = Path(args.project_path).expanduser().resolve()
    if selected_path.is_dir():
        project_dir = selected_path
    elif selected_path.is_file() and selected_path.name.casefold() == "project.yaml":
        project_dir = selected_path.parent
    elif (
        selected_path.is_file()
        and selected_path.name.casefold() == "other.yaml"
        and selected_path.parent.name.casefold() == "configs"
    ):
        project_dir = selected_path.parent.parent
    else:
        parser.error(
            "project path must be a project directory, its project.yaml, "
            "or configs/other.yaml"
        )

    project_config_path = project_dir / "project.yaml"
    other_config_path = project_dir / "configs" / "other.yaml"
    try:
        if not project_config_path.is_file():
            raise FileNotFoundError(f"Project config does not exist: {project_config_path}")
        if not other_config_path.is_file():
            raise FileNotFoundError(f"Training config does not exist: {other_config_path}")

        from animalposetracker.project.model_context import (
            configure_project_model_spec,
            load_project_context,
            read_yaml_mapping,
        )

        project_values = read_yaml_mapping(project_config_path)
        required_fields = ("project_name", "worker", "model_type", "model_scale", "date")
        missing_fields = [
            name for name in required_fields
            if not str(project_values.get(name) or "").strip()
        ]
        if missing_fields:
            raise ValueError(
                f"Project config is missing required fields: {', '.join(missing_fields)}"
            )
        config, context = load_project_context(other_config_path)
        _model_spec, _head_name, _scale = configure_project_model_spec(config, context)
    except (OSError, TypeError, ValueError, NotImplementedError, yaml.YAMLError) as exc:
        parser.error(f"Could not load project at {project_dir}: {exc}")

    dataset_names = context["dataset"].get("names") or {}
    class_count = len(dataset_names) if isinstance(dataset_names, (dict, list, tuple)) else "?"
    keypoint_shape = context["dataset"].get("kpt_shape", "unknown")
    model_name = str(project_values.get("model_type") or "model")
    model_scale = str(project_values.get("model_scale") or "")
    print(f"Project loaded and validated: {project_dir}")
    print(f"Model: {model_name}-{model_scale}".rstrip("-"))
    print(f"Model config: {config.model}")
    print(f"Dataset config: {config.data}")
    print(f"Classes: {class_count}")
    print(f"Keypoints: {keypoint_shape}")
    return 0


def run(argv=None) -> int:
    args = _make_create_parser().parse_args(argv)
    _create_project(args)
    return 0
