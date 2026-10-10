"""Command-line project creation and configuration."""

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from animalposetracker.cfg import DATA_YAML_PATHS
from .api import AnimalPoseTrackerProject


def _make_create_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument(
        "--pretrained",
        nargs="?",
        const=True,
        metavar="PATH",
        help=(
            "Start from pretrained weights (default). Optionally record a local "
            "checkpoint path in configs/other.yaml; nothing is checked or "
            "downloaded at create time."
        ),
    )
    pretrained.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train from scratch.",
    )
    parser.set_defaults(pretrained=True)
    return parser


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
    project.dataset_config.update(dataset_config)
    project.dataset_config["names"] = class_names
    project.project_config.update({
        "classes": len(class_names),
        "classes_name": ordered_names,
        "keypoints": keypoint_count,
        "keypoints_name": keypoint_names,
        "visible": (keypoint_dims == 3),
        "skeleton": dataset_config.get("skeleton", []),
        "project_path": str(project.project_path),
    })
    project._sync_model_config_from_dataset()
    project.other_config.update({
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "pretrained": args.pretrained,
        "val": True,
        "plots": True,
        "project": "runs",
        "name": "train",
        "exist_ok": False,
    })
    # Pretrained weight resolution is deferred to training time so project
    # creation never performs file checks or network downloads.
    project.save_configs("all")
    _print_create_banner(args, project, dataset_root)
    return project


def _print_create_banner(args, project: AnimalPoseTrackerProject, dataset_root: Path) -> None:
    from animalposetracker.training.console import (
        describe_pretrained,
        environment_line,
        format_summary_block,
    )

    class_names = project.project_config.get("classes_name") or []
    keypoints = project.project_config.get("keypoints")
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
            f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
            f"workers={args.workers} device={args.device}",
        ),
        ("Pretrained", describe_pretrained(args.pretrained)),
    ]))
    print()
    print(f"Next: animalpose-cli train --config {project.project_path / 'configs' / 'other.yaml'}")


def run(argv=None) -> int:
    args = _make_create_parser().parse_args(argv)
    _create_project(args)
    return 0
