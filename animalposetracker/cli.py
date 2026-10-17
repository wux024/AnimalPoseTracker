"""Unified command dispatcher for AnimalPoseTracker workflows."""

import argparse
import sys
from typing import Optional, Sequence

from animalposetracker import __version__


COMMANDS = {
    "create": "Create and configure a project",
    "load": "Validate and display an existing project",
    "train": "Train a pose model",
    "val": "Validate a checkpoint on the configured validation split",
    "predict": "Predict on images, videos, or a dataset split",
    "infer": "Run live inference on a camera, video file, or stream URL",
    "export": "Export a checkpoint to a deployment format",
    "pose3d": "Triangulate multi-view 2D predictions into 3D keypoints",
}


def _make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli",
        description="Create and load projects, then train, validate, predict, and export models.",
        epilog=(
            "The separate `animalposetracker` command opens the GUI. Each workflow accepts "
            "its own options; use `animalpose-cli <command> --help` for details."
        ),
    )
    parser.add_argument("--version", action="version", version=f"animalposetracker {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    for command, help_text in COMMANDS.items():
        subparsers.add_parser(command, help=help_text, add_help=False)
    return parser


def _run_validation(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli val",
        description="Validate a pose checkpoint on the configured validation split.",
        epilog=(
            "Validation accepts the same config override options as `train`, including "
            "--batch-size, --image-size, --num-workers, --device, "
            "--validation-confidence, --validation-iou, and --max-detections."
        ),
    )
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir")
    values, overrides = parser.parse_known_args(list(argv))

    from animalposetracker.training.cli import run

    forwarded = [
        "--validate-only", "--config", values.config, "--weights", values.weights,
    ]
    if values.output_dir:
        forwarded.extend(("--output-dir", values.output_dir))
    forwarded.extend(overrides)
    return int(run(forwarded) or 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to an existing workflow CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _make_argument_parser()
    if not arguments:
        parser.print_help()
        return 0

    parsed, forwarded = parser.parse_known_args(arguments)
    if parsed.command is None:
        parser.error(f"unknown command or option: {arguments[0]}")

    command_args = list(forwarded)
    if parsed.command == "train":
        from animalposetracker.training.cli import run

    elif parsed.command == "val":
        return _run_validation(command_args)
    elif parsed.command == "predict":
        from animalposetracker.prediction.cli import run

    elif parsed.command == "create":
        from animalposetracker.project.cli import run

    elif parsed.command == "load":
        from animalposetracker.project.cli import run_load

        return int(run_load(command_args) or 0)

    elif parsed.command == "infer":
        from animalposetracker.inference.cli import run

    elif parsed.command == "pose3d":
        from animalposetracker.pose3d.cli import run

        return int(run(command_args) or 0)

    else:
        from animalposetracker.export.cli import run

    return int(run(command_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
