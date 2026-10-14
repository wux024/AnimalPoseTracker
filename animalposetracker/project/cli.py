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
        epilog=(
            "Short spellings (--batch, --imgsz, --lr0, ...) remain accepted as "
            "deprecated aliases; the full names are canonical."
        ),
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
        value_type = field_types[name]
        if name == "scale":
            parser.add_argument(
                "--augment-scale", dest=dest, type=value_type, default=argparse.SUPPRESS,
                help="Training scale augmentation; --scale selects the model scale.",
            )
            continue
        primary = f"--{name.replace('_', '-')}"
        aliases = _CLI_ALIASES.get(name, ())
        if value_type is bool:
            _add_bool_options(parser, (primary,), dest)
            for alias in aliases:
                _add_bool_options(parser, (alias,), dest, deprecated_for=primary)
        else:
            parser.add_argument(primary, dest=dest, type=value_type, default=argparse.SUPPRESS)
            for alias in aliases:
                parser.add_argument(
                    alias, dest=dest, type=value_type, default=argparse.SUPPRESS,
                    help=f"deprecated alias of {primary}",
                )

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
            "checkpoint path in configs/runtime.yaml; nothing is checked or "
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
        help="Record a local checkpoint path in configs/runtime.yaml.",
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
            values[_CLI_FIELD_TO_KEY[name]] = getattr(args, dest)
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
    # The template's left/right swap table has to travel with the project: without it the
    # derived dataset.yaml keeps the identity mapping, and fliplr then mirrors keypoint
    # coordinates without swapping their left/right labels.
    template_flip = dataset_config.get("flip_idx")
    if (
        isinstance(template_flip, (list, tuple))
        and len(template_flip) == keypoint_count
        and sorted(int(index) for index in template_flip) == list(range(keypoint_count))
    ):
        project.dataset_config["flip_idx"] = [int(index) for index in template_flip]
    project.dataset_config["path"] = str(dataset_root)
    # Defaults the create flow pins for this project's training runs.
    project.set_runtime_value("train.shared.runtime.validation_enabled", True)
    project.set_runtime_value("train.shared.runtime.plots", True)
    project.set_runtime_value("common.project", "runs")
    project.set_runtime_value("common.name", "train")
    project.set_runtime_value("common.exist_ok", False)
    project.apply_runtime_overrides(_collect_create_training_overrides(args))
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

    class_names = list((project.dataset_config.get("names") or {}).values())
    kpt_shape = project.dataset_config.get("kpt_shape") or [None]
    keypoints = kpt_shape[0]
    train_tree = project.runtime_config.get("train") or {}
    runtime_shared = (train_tree.get("shared") or {}).get("runtime") or {}
    model_type = str(project.project_config.get("model_type") or "AnimalRTPose")
    profile = ((train_tree.get("models") or {}).get(model_type) or {}).get("training") or {}
    common = project.runtime_config.get("common") or {}
    pretrained = runtime_shared.get("pretrained", False)
    if runtime_shared.get("pretrained_weights") and pretrained is not False:
        pretrained = runtime_shared["pretrained_weights"]
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
            f"epochs={profile.get('epochs', '?')} "
            f"batch={profile.get('batch_size', '?')} "
            f"imgsz={profile.get('image_size', '?')} "
            f"workers={runtime_shared.get('num_workers', '?')} "
            f"device={common.get('device', '?')}",
        ),
        ("Pretrained", describe_pretrained(pretrained)),
    ]))
    print()
    print(f"Next: animalpose-cli train --config {project.project_path / 'configs' / 'runtime.yaml'}")


def _make_load_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli load",
        description="Validate and display an existing AnimalPoseTracker project without opening the GUI.",
    )
    parser.add_argument(
        "project_path",
        help="Project directory, project.yaml, or configs/runtime.yaml",
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
        and selected_path.name.casefold() == "runtime.yaml"
        and selected_path.parent.name.casefold() == "configs"
    ):
        project_dir = selected_path.parent.parent
    else:
        parser.error(
            "project path must be a project directory, its project.yaml, "
            "or configs/runtime.yaml"
        )

    project_config_path = project_dir / "project.yaml"
    runtime_config_path = project_dir / "configs" / "runtime.yaml"
    try:
        if not project_config_path.is_file():
            raise FileNotFoundError(f"Project config does not exist: {project_config_path}")
        if not runtime_config_path.is_file():
            raise FileNotFoundError(f"Runtime config does not exist: {runtime_config_path}")

        from animalposetracker.project.model_context import (
            configure_project_model_spec,
            load_project_runtime_context,
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
        config, context = load_project_runtime_context(project_dir)
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
    # Which values actually differ from the package defaults, so "why is it
    # X" has an answer without reading the YAML (the project snapshot carries
    # every registered key; only real changes are listed).
    resolved = context.get("resolved")
    if resolved is not None:
        changed = resolved.changed_from_defaults()
        if changed:
            print("Values that differ from the package defaults:")
            for path, value in sorted(changed.items()):
                print(f"  {path} = {value}")
    return 0


def run(argv=None) -> int:
    args = _make_create_parser().parse_args(argv)
    _create_project(args)
    return 0
