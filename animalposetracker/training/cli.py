"""Command-line worker for AnimalPoseTracker's own training process."""

import argparse
import json
import os
import signal
import socket
import sys
import subprocess
import shutil
import time
import threading
import traceback
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from .batch_probe import parse_batch_size, probe_batch_size
from .config import TrainingConfig
from .events import EventEmitter, TrainingEvent
from .profiles import (
    ANIMALVITPOSE_AUGMENTATION,
    ANIMALVITPOSE_PREPROCESSING,
    ANIMALVITPOSE_VALIDATION,
    KEYPOINT_METRICS,
    apply_animalvitpose_recipe,
    flatten_train_values,
    train_key_targets,
)
from animalposetracker.config import project_dir_from, resolve_scopes
from animalposetracker.nn.model_config import head_config, set_simcc_keypoint_count


def _read_mapping(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _resolve_project_pretrained_path(source, project_dir: Path) -> Path:
    """Cache downloaded initialization weights inside their owning project."""
    from .checkpoint import resolve_checkpoint_path

    cache_dir = Path(project_dir) / "pretrained"
    source_text = str(source)
    if source_text.lower().startswith(("http://", "https://")):
        filename = Path(urlparse(source_text).path).name
        shared_cache = Path.home() / ".cache" / "animalposetracker" / "pretrained" / filename
        destination = cache_dir / filename
        destination_ready = destination.is_file() and destination.stat().st_size > 0
        if (
            filename
            and not destination_ready
            and shared_cache.is_file()
            and shared_cache.stat().st_size > 0
        ):
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_dir / f".{filename}.{os.getpid()}.cache-copy"
            try:
                shutil.copy2(shared_cache, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            return destination.resolve()
    return resolve_checkpoint_path(source, cache_dir=cache_dir)


def _resolve_project_path(value, project_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


_FULL_MODEL_PRETRAINED_TYPES = frozenset({"AnimalRTPose"})


def _resolve_project_default_pretrained(project_dir: Path, project_values: Dict[str, Any]) -> Path:
    """Locate the bundled full-model checkpoint for ``pretrained: true`` projects.

    Resolution happens at training time so project directories stay movable and
    ``animalpose-cli create`` never performs file checks or network downloads.
    AnimalViTPose is excluded: its backbone initialization is resolved by the
    training profile instead of a full-model checkpoint.
    """
    model_type = project_values.get("model_type")
    model_scale = project_values.get("model_scale")
    filename = f"{model_type}-{model_scale}.pt".lower()
    candidate = (Path(project_dir) / "pretrained" / filename).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found: {candidate}\n"
            "Place the checkpoint at that path, record an explicit path in "
            "configs/runtime.yaml (pretrained: /path/to/checkpoint.pt), or set "
            "pretrained: false to train from scratch."
        )
    return candidate


def _print_train_banner(config, project_values: Dict[str, Any], validate_only: bool) -> None:
    """Ultralytics-style startup banner for interactive terminals."""
    from ..console import environment_line, format_summary_block

    model_type = str(project_values.get("model_type") or "model")
    model_scale = str(project_values.get("model_scale") or "")
    rows = [
        ("Model", f"{model_type}-{model_scale}".rstrip("-")),
        ("Model cfg", str(config.model)),
        ("Data", str(config.data)),
        (
            "Training",
            f"epochs={config.epochs} batch={config.batch_size} imgsz={config.image_size} "
            f"workers={config.num_workers} device={config.device}",
        ),
        ("Optimizer", f"{config.optimizer} lr={config.learning_rate} momentum={config.momentum}"),
        ("Output", str(config.output_dir)),
        (
            "Pretrained",
            str(config.pretrained_weights) if config.pretrained_weights else "disabled (train from scratch)",
        ),
    ]
    print()
    print(environment_line())
    print()
    if validate_only:
        print("  Mode  validate-only")
        print()
    print(format_summary_block(rows))
    print()


def _load_evaluation_weights(weights_path, trainer, emitter):
    """Load a full AnimalPoseTracker checkpoint or compatible weights-only file."""
    from .checkpoint import load_checkpoint, load_model_weights, resolve_checkpoint_path

    path = resolve_checkpoint_path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation weights do not exist: {path}")
    try:
        metadata = load_checkpoint(
            path,
            trainer.raw_model,
            map_location=trainer.device,
            restore_rng=False,
            ema_model=trainer.ema.model if trainer.ema.enabled else None,
        )
        source_kind = "AnimalPoseTracker checkpoint"
        loaded_tensors = len(trainer.raw_model.state_dict())
    except ValueError as exc:
        if "Not an AnimalPoseTracker training checkpoint" not in str(exc):
            raise
        report = load_model_weights(path, trainer.raw_model, map_location=trainer.device, strict=False)
        if trainer.ema.enabled:
            trainer.ema.model.load_state_dict(trainer.raw_model.state_dict())
        source_kind = "weights-only checkpoint"
        loaded_tensors = int(report["loaded_tensors"])
        metadata = {"epoch": None, "format_version": None}

    emitter.emit(TrainingEvent(
        event="evaluation_weights",
        metrics={
            "loaded_tensors": float(loaded_tensors),
            "epoch": float(metadata["epoch"]) if metadata.get("epoch") is not None else -1.0,
        },
        message=f"Loaded {source_kind}: {path}",
    ))
    return path


def _device_ids(device) -> list:
    if isinstance(device, (list, tuple)):
        parts = [str(item).strip() for item in device]
    else:
        value = str(device or "auto").strip().lower()
        if value.startswith("cuda:"):
            value = value.split(":", 1)[1]
        parts = [part.strip() for part in value.split(",")]
    return parts if len(parts) > 1 and all(part.isdigit() for part in parts) else []


def _launch_distributed(
    config_path: Path,
    resume_path: str = None,
    output_dir: str = None,
    device=None,
    cli_overrides=None,
) -> int:
    """Launch one local training process per selected GPU with torch.distributed."""
    import torch

    project_dir = project_dir_from(config_path)
    # Only the device is needed here; it lives in the shared section.
    config_values = resolve_scopes(project_dir, ("common",)).flat_for("common")
    device_ids = _device_ids(device if device is not None else config_values.get("device"))
    if not torch.cuda.is_available():
        raise RuntimeError("Multiple GPU devices were selected, but this PyTorch environment has no CUDA")
    if len(device_ids) > torch.cuda.device_count():
        raise RuntimeError(
            f"Selected {len(device_ids)} GPUs, but PyTorch can see only {torch.cuda.device_count()}"
        )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
    # PyTorch's env:// rendezvous supports the classic TCPStore on builds without libuv.
    environment["USE_LIBUV"] = "0"
    environment.setdefault("OMP_NUM_THREADS", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        master_port = probe.getsockname()[1]
    children = []
    for rank in range(len(device_ids)):
        worker_environment = environment.copy()
        worker_environment.update({
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(len(device_ids)),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        })
        worker_environment[_CLI_OVERRIDE_ENV] = json.dumps(cli_overrides or {})
        command = [
            sys.executable,
            "-m",
            "animalposetracker.training.cli",
            "--config",
            str(config_path),
        ]
        if output_dir:
            command.extend(("--output-dir", str(Path(output_dir).expanduser().resolve())))
        if resume_path:
            command.extend(("--resume", str(Path(resume_path).expanduser().resolve())))
        children.append(subprocess.Popen(command, cwd=str(config_path.parent.parent), env=worker_environment))
    try:
        while True:
            exit_codes = [child.poll() for child in children]
            failures = [code for code in exit_codes if code not in (None, 0)]
            if failures:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    child.wait()
                return failures[0]
            if all(code is not None for code in exit_codes):
                return 0
            time.sleep(0.1)
    except KeyboardInterrupt:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        return 130


_CLI_OVERRIDE_ENV = "ANIMALPOSE_TRAIN_OVERRIDES"


def _parse_cache_value(value: str):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", "none"}:
        return False
    if normalized in {"ram", "disk"}:
        return normalized
    raise argparse.ArgumentTypeError("cache must be false, true, 'ram', or 'disk'")


def _parse_pretrained_value(value: str):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


def _parse_multi_scale(value: str) -> float:
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "on"}:
        return 1.0
    if normalized in {"false", "no", "off"}:
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("multi_scale must be a number from 0 to 1") from exc


def _parse_freeze(value: str):
    try:
        items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("freeze must be a layer count or comma-separated indices") from exc
    if len(items) == 1:
        return items[0]
    if items:
        return items
    raise argparse.ArgumentTypeError("freeze must be a layer count or comma-separated indices")


def _parse_int_list(value: str) -> List[int]:
    try:
        return [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers") from exc

_CLI_ALIASES = {
    "batch_size": ("--batch",),
    "image_size": ("--imgsz",),
    "num_workers": ("--workers",),
    "time_hours": ("--time",),
    "learning_rate": ("--lr0",),
    "final_lr_factor": ("--lrf",),
    "cosine_schedule": ("--cos-lr",),
    "nominal_batch_size": ("--nbs",),
    "validation_batch_size": ("--val-batch",),
    "validation_confidence": ("--conf",),
    "validation_iou": ("--iou",),
    "max_detections": ("--max-det",),
    "early_stopping_patience": ("--patience",),
    "box_loss_weight": ("--box",),
    "class_loss_weight": ("--cls",),
    "distribution_loss_weight": ("--dfl",),
    "keypoint_loss_weight": ("--pose",),
    "visibility_loss_weight": ("--kobj",),
    "scale": ("--augment-scale",),
}

# Field name -> canonical key, only where the two spellings differ.
_CLI_FIELD_TO_CANONICAL = {
    "validation_confidence": "confidence",
    "max_detections": "max_det",
}

_CLI_INTERNAL_FIELDS = {
    "project_dir", "output_dir", "resume_from", "train_dataset_size",
    "resolved_optimizer", "resolved_learning_rate", "resolved_momentum", "resolved_weight_decay",
}
_CLI_COMMAND_FIELDS = tuple(
    item.name for item in dataclass_fields(TrainingConfig)
    if item.name not in _CLI_INTERNAL_FIELDS
)
_CLI_FIELD_TO_KEY = {
    name: _CLI_FIELD_TO_CANONICAL.get(name, name)
    for name in _CLI_COMMAND_FIELDS
}
_CLI_SPECIAL_TYPES = {
    "model": str,
    "data": str,
    "batch_size": parse_batch_size,
    "pretrained_weights": str,
    "time_hours": float,
    "validation_batch_size": int,
    "channels_last": bool,
    "cache": _parse_cache_value,
    "multi_scale": _parse_multi_scale,
    "freeze": _parse_freeze,
    "warmup_iters": int,
    "warmup_start_factor": float,
    "gradient_clip_norm": float,
    "layer_decay_rate": float,
    "lr_milestones": _parse_int_list,
}


def _add_bool_options(parser: argparse.ArgumentParser, flags, dest: str, deprecated_for: str = None) -> None:
    help_text = f"deprecated alias of {deprecated_for}" if deprecated_for else None
    for flag in flags:
        # A bare switch: the flag is on, `--no-` turns it off, and leaving both out keeps the
        # value unresolved so the config file still decides. Accepting `--flag true` as well
        # made the help read `[--flag [FLAG]]`, which suggests a value is expected.
        parser.add_argument(flag, dest=dest, action="store_true",
                            default=argparse.SUPPRESS, help=help_text)
        negative = "--no-" + flag[2:]
        parser.add_argument(negative, dest=dest, action="store_false",
                            default=argparse.SUPPRESS, help=help_text)


def _collect_cli_overrides(args):
    """The values this command line spelled out, in canonical keys.

    Returns ``(values, fields)``: ``values`` keyed by canonical name (what the
    resolver speaks) and ``fields`` keyed by the config object's field names
    (what the resume whitelist speaks).
    """
    values = {}
    for field_name in _CLI_COMMAND_FIELDS:
        if hasattr(args, field_name):
            values[_CLI_FIELD_TO_KEY[field_name]] = getattr(args, field_name)
    for key in ("project", "name", "exist_ok"):
        if hasattr(args, key):
            values[key] = getattr(args, key)
    if hasattr(args, "pretrained"):
        values["pretrained"] = getattr(args, "pretrained")
        if args.pretrained is False:
            values["pretrained_weights"] = None
        elif isinstance(args.pretrained, str):
            values["pretrained_weights"] = args.pretrained
    if hasattr(args, "pretrained_weights"):
        values["pretrained_weights"] = args.pretrained_weights

    fields = {
        field_name: values[key]
        for field_name, key in _CLI_FIELD_TO_KEY.items()
        if key in values
    }
    return values, fields


def _build_train_overrides(cli_values, model_type):
    """Anchor flat command-line values on the registry's dotted paths.

    The path is the one the flatten actually reads the value from, so an
    override lands where it decides -- ``seed`` on the ViTPose recipe, for
    instance, not on the shared default it would shadow.
    """
    targets = train_key_targets(model_type)
    overrides = {}
    for name, value in cli_values.items():
        if name in ("device", "project", "name", "exist_ok"):
            overrides[f"common.{name}"] = value
            continue
        if value == "auto":
            # Probed later in the parent (single GPU) or before the launch (multi GPU);
            # the resolver must never see the placeholder.
            continue
        path = targets.get(name)
        if path is not None:
            overrides[path] = value
    return overrides


def _inherited_override_paths() -> Dict[str, Any]:
    """Overrides handed to a distributed child process (dotted paths)."""
    inherited = os.environ.get(_CLI_OVERRIDE_ENV)
    if not inherited:
        return {}
    values = json.loads(inherited)
    if not isinstance(values, dict):
        raise ValueError(f"{_CLI_OVERRIDE_ENV} must contain a JSON mapping")
    return {str(key): value for key, value in values.items()}


def _make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli train",
        description="Train a pose model with AnimalPoseTracker's PyTorch training engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Short spellings (--batch, --imgsz, --lr0, ...) remain accepted as "
            "deprecated aliases; the full names are canonical."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/runtime.yaml",
        help="Project directory or its configs/runtime.yaml (new-format projects).",
    )
    parser.add_argument(
        "--resume", "--resume-from", dest="resume",
        help="Resume from an AnimalPoseTracker full training checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run the configured validation split without training.",
    )
    parser.add_argument("--weights", help="Checkpoint or weights-only file used by --validate-only.")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir")
    parser.add_argument("--json", action="store_true", help="Emit JSON-lines events on stdout.")
    parser.add_argument("--project", default=argparse.SUPPRESS, help="Run output root (project setting).")
    parser.add_argument("--name", default=argparse.SUPPRESS, help="Run name.")
    parser.add_argument("--exist-ok", dest="exist_ok", action="store_const", const=True, default=argparse.SUPPRESS)
    parser.add_argument("--no-exist-ok", dest="exist_ok", action="store_const", const=False, default=argparse.SUPPRESS)

    for item in dataclass_fields(TrainingConfig):
        name = item.name
        if name in _CLI_INTERNAL_FIELDS or name in {"cache", "pretrained_weights"}:
            continue
        primary = f"--{name.replace('_', '-')}"
        aliases = _CLI_ALIASES.get(name, ())
        value_type = _CLI_SPECIAL_TYPES.get(name, type(item.default))
        if value_type is bool:
            _add_bool_options(parser, (primary,), name)
            for alias in aliases:
                _add_bool_options(parser, (alias,), name, deprecated_for=primary)
        else:
            parser.add_argument(primary, dest=name, type=value_type, default=argparse.SUPPRESS)
            for alias in aliases:
                parser.add_argument(
                    alias, dest=name, type=value_type, default=argparse.SUPPRESS,
                    help=f"deprecated alias of {primary}",
                )

    parser.add_argument("--cache", nargs="?", const=True, type=_parse_cache_value, default=argparse.SUPPRESS)
    parser.add_argument("--no-cache", dest="cache", action="store_const", const=False, default=argparse.SUPPRESS)
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument(
        "--pretrained", nargs="?", const=True, type=_parse_pretrained_value,
        default=argparse.SUPPRESS, metavar="PATH",
        help="Use pretrained weights, optionally naming a local checkpoint.",
    )
    pretrained.add_argument("--no-pretrained", dest="pretrained", action="store_const", const=False, default=argparse.SUPPRESS)
    pretrained.add_argument("--pretrained-weights", dest="pretrained_weights", default=argparse.SUPPRESS)
    parser.add_argument("--no-gradient-clip", dest="gradient_clip_norm", action="store_const", const=None, default=argparse.SUPPRESS)
    parser.add_argument("--quiet", dest="verbose", action="store_const", const=False, default=argparse.SUPPRESS)
    return parser


def run(argv=None) -> int:
    args = _make_argument_parser().parse_args(argv)
    cli_override_values, cli_override_fields = _collect_cli_overrides(args)
    pretty_console = not (args.json or not sys.stdout.isatty())
    renderer = None
    if pretty_console:
        from .console import PrettyTrainingRenderer, loss_columns

        renderer = PrettyTrainingRenderer()
        emitter = EventEmitter(callback=renderer.emit)
    else:
        emitter = EventEmitter.json_stdout()
    trainer = None
    event_stream = None
    distributed_initialized = False
    rank = 0
    try:
        config_path = Path(args.config).expanduser().resolve()
        project_dir = project_dir_from(config_path)
        project_config_path = project_dir / "project.yaml"
        project_values = _read_mapping(project_config_path) if project_config_path.is_file() else {}
        model_type = str(project_values.get("model_type") or "AnimalRTPose")

        overrides = {
            **_inherited_override_paths(),
            **_build_train_overrides(cli_override_values, model_type),
        }
        resolved = resolve_scopes(project_dir, ("common", "train"), overrides=overrides)
        values, _origins = flatten_train_values(resolved, model_type)
        if values.get("model") is None:
            values["model"] = str(project_dir / "configs" / "model.yaml")
        if values.get("data") is None:
            values["data"] = str(project_dir / "configs" / "dataset.yaml")
        if args.validate_only:
            if args.resume:
                raise ValueError("--resume cannot be combined with --validate-only")
            if not args.weights:
                raise ValueError("--validate-only requires --weights")
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise ValueError("Validation-only mode currently requires a single process")
            values["validation_enabled"] = True
            values["pretrained"] = False
            values["pretrained_weights"] = None
            values["resume"] = False
            values["resume_from"] = None
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        if args.output_dir:
            from animalposetracker.project.model_context import unique_output_directory

            output_path = Path(args.output_dir).expanduser()
            if not output_path.is_absolute():
                output_path = project_dir / output_path
            if world_size_env <= 1:
                output_path = unique_output_directory(
                    output_path, exist_ok=bool(values.get("exist_ok", False))
                )
            values["output_dir"] = str(output_path.resolve())
        elif args.validate_only and world_size_env <= 1:
            from animalposetracker.project.model_context import unique_output_directory

            validation_root = project_dir / "results" / "val"
            values["output_dir"] = str(unique_output_directory(
                validation_root,
                exist_ok=bool(values.get("exist_ok", False)),
            ))
        elif not args.validate_only and not args.resume and world_size_env <= 1:
            from animalposetracker.project.model_context import unique_output_directory

            configured_output = values.get("output_dir", values.get("save_dir"))
            if configured_output is None:
                run_root = Path(str(values.get("project") or "results")).expanduser()
                if not run_root.is_absolute():
                    run_root = project_dir / run_root
                output_path = run_root / str(values.get("name") or "train")
            else:
                output_path = Path(str(configured_output)).expanduser()
                if not output_path.is_absolute():
                    output_path = project_dir / output_path
            values["output_dir"] = str(unique_output_directory(
                output_path,
                exist_ok=bool(values.get("exist_ok", False)),
            ))
        elif not args.validate_only and not args.resume and world_size_env > 1:
            configured_output = values.get("output_dir", values.get("save_dir"))
            if configured_output is None:
                run_root = Path(str(values.get("project") or "results")).expanduser()
                if not run_root.is_absolute():
                    run_root = project_dir / run_root
                output_path = run_root / str(values.get("name") or "train")
            else:
                output_path = Path(str(configured_output)).expanduser()
                if not output_path.is_absolute():
                    output_path = project_dir / output_path
            if output_path.exists() and not bool(values.get("exist_ok", False)):
                raise FileExistsError(
                    f"Training output already exists: {output_path}. Choose a new name or set exist_ok=true."
                )
            values["output_dir"] = str(output_path.resolve())
        if world_size_env <= 1 and len(_device_ids(values.get("device"))) > 1:
            if args.validate_only:
                device_ids = _device_ids(values.get("device"))
                values["device"] = device_ids[0]
                emitter.emit(TrainingEvent(
                    event="evaluation_device",
                    message=f"Validation uses the first selected device ({device_ids[0]}).",
                ))
            else:
                if values.get("batch_size") == "auto":
                    values["batch_size"] = probe_batch_size(
                        data_yaml=values["data"],
                        model_yaml=values["model"],
                        model_scale=str(project_values.get("model_scale") or "n").lower(),
                        image_size=int(values["image_size"]),
                        device=str(_device_ids(values.get("device"))[0]),
                    )
                    targets = train_key_targets(model_type)
                    overrides = {
                        **overrides,
                        targets["batch_size"]: str(values["batch_size"]),
                    }
                    emitter.emit(TrainingEvent(
                        event="batch_size_probe",
                        message=(
                            f"--batch-size auto probed {values['batch_size']} per GPU "
                            f"(largest that fits 60% of the GPU memory)."
                        ),
                    ))
                return _launch_distributed(
                    config_path,
                    args.resume,
                    output_dir=values.get("output_dir"),
                    device=values.get("device"),
                    cli_overrides=overrides,
                )
        resume_config = None
        if args.resume:
            values["resume_from"] = args.resume
            values["output_dir"] = str(Path(args.resume).expanduser().resolve().parent.parent)
            values["pretrained_weights"] = None
            values["pretrained"] = False
            from dataclasses import fields
            from .checkpoint import read_checkpoint_config

            saved_values = read_checkpoint_config(args.resume)
            if saved_values is not None:
                current_config = TrainingConfig.from_mapping(values, project_dir=project_dir)
                allowed_fields = {field.name for field in fields(TrainingConfig)}
                saved_config = TrainingConfig(**{
                    key: value for key, value in saved_values.items()
                    if key in allowed_fields
                })
                # Ultralytics restores the checkpoint's arguments and allows these
                # runtime overrides when resuming.
                for name in ("image_size", "batch_size", "device", "close_mosaic"):
                    setattr(saved_config, name, getattr(current_config, name))
                for name in cli_override_fields:
                    if hasattr(saved_config, name):
                        setattr(saved_config, name, getattr(current_config, name))
                saved_config.output_dir = current_config.output_dir
                saved_config.resume_from = current_config.resume_from
                saved_config.pretrained_weights = None
                saved_config.project_dir = current_config.project_dir
                if saved_config.model is None or not saved_config.model.is_file():
                    saved_config.model = current_config.model
                if saved_config.data is None or not saved_config.data.is_file():
                    saved_config.data = current_config.data
                resume_config = saved_config
                values["model"] = str(saved_config.model) if saved_config.model else None
                values["data"] = str(saved_config.data) if saved_config.data else None
        scale = str(project_values.get("model_scale") or "n").lower()
        model_spec = None
        head_name = None
        profile_defaults = []
        if values.get("model"):
            candidate_model_path = _resolve_project_path(values["model"], project_dir)
            if candidate_model_path.is_file():
                model_spec = _read_mapping(candidate_model_path)
                head_name, _head_entry = head_config(model_spec)
                if head_name == "SimCCHead":
                    values, profile_defaults = apply_animalvitpose_recipe(
                        values,
                        model_scale=scale,
                        model_config=model_spec,
                    )
        if args.validate_only:
            values["validation_enabled"] = True
            values["pretrained"] = False
            values["pretrained_weights"] = None
            values["resume"] = False
            values["resume_from"] = None
        from .engine import initialize_distributed, seed_everything

        rank, world_size, _local_rank = initialize_distributed(values.get("device") or "auto")
        distributed_initialized = world_size > 1
        if rank != 0:
            emitter = EventEmitter()
        if (
            values.get("pretrained") is True
            and not values.get("pretrained_weights")
            and str(project_values.get("model_type")) in _FULL_MODEL_PRETRAINED_TYPES
        ):
            values["pretrained_weights"] = str(
                _resolve_project_default_pretrained(project_dir, project_values)
            )
        cli_requested_auto = str(getattr(args, "batch_size", "")).strip().lower() == "auto"
        if cli_requested_auto or values.get("batch_size") == "auto":
            values["batch_size"] = probe_batch_size(
                data_yaml=values["data"],
                model_yaml=values["model"],
                model_scale=str(project_values.get("model_scale") or "n").lower(),
                image_size=int(values["image_size"]),
                device=str(values.get("device") or "auto"),
            )
            emitter.emit(TrainingEvent(
                event="batch_size_probe",
                message=(
                    f"--batch-size auto probed a batch size of {values['batch_size']} "
                    f"(largest that fits 60% of the GPU memory)."
                ),
            ))
        config = resume_config or TrainingConfig.from_mapping(values, project_dir=project_dir)
        rank_seed = config.seed + (1 + rank if world_size > 1 else 0)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            event_stream = (config.output_dir / "training.jsonl").open("a", encoding="utf-8")
            emitter.add_stream(event_stream)
        if config.model is None or config.data is None:
            raise ValueError("The project configuration must include model and data paths")
        if not config.model.is_file():
            raise FileNotFoundError(f"Model configuration does not exist: {config.model}")
        if not config.data.is_file():
            raise FileNotFoundError(f"Dataset configuration does not exist: {config.data}")

        if rank == 0 and pretty_console:
            _print_train_banner(config, project_values, args.validate_only)

        if rank == 0 and profile_defaults:
            emitter.emit(TrainingEvent(
                event="model_profile",
                message=(
                    f"AnimalViTPose ({scale}) MMPose defaults applied for: "
                    + ", ".join(profile_defaults)
                    + ". Explicit non-default project settings are preserved."
                ),
            ))

        if model_spec is None:
            model_spec = _read_mapping(config.model)
            head_name, _head_entry = head_config(model_spec)

        if renderer is not None:
            # The progress row's loss columns follow the head: an AnimalViTPose run has no
            # box/pose/kobj/cls/dfl components to show, and its SimCC losses instead.
            renderer.columns = loss_columns(head_name)

        if float(values.get("copy_paste", 0.0) or 0.0) > 0:
            raise NotImplementedError(
                "copy_paste is a segmentation-only augmentation in this trainer; set it to 0 for pose training"
            )

        from animalposetracker.nn import build_model
        from animalposetracker.nn.head import SimCCHead, YOLOPoseHead
        from .checkpoint import load_model_weights
        from animalposetracker.data.pose import build_pose_dataloaders
        from .engine import Trainer
        from .losses import PoseDetectionLoss
        from animalposetracker.evaluation.animalrtpose import PoseDetectionValidator

        seed_everything(rank_seed, config.deterministic)
        if config.single_cls:
            model_spec["nc"] = 1

        if config.batch_size % world_size:
            raise ValueError(
                f"Global batch_size={config.batch_size} must be divisible by world_size={world_size}"
            )
        if (
            config.validation_batch_size is not None
            and config.validation_batch_size % world_size
        ):
            raise ValueError(
                f"validation_batch_size={config.validation_batch_size} must be divisible "
                f"by world_size={world_size}"
            )
        # Start each process's data-loader RNG stream at a distinct seed.
        seed_everything(rank_seed, config.deterministic)
        simcc_input_size = None
        simcc_split_ratio = 2.0
        if head_name == "SimCCHead":
            from .profiles import configure_animalvitpose_model

            variant_name = configure_animalvitpose_model(
                model_spec,
                scale=scale,
                image_size=config.image_size,
            )
            _module_name, head_entry = head_config(model_spec)
            head_args = head_entry[3]
            if not isinstance(head_args, (list, tuple)) or len(head_args) < 3:
                raise ValueError("SimCCHead requires output channels, input_size and feature-map size")
            simcc_input_size = tuple(map(int, head_args[1]))
            simcc_split_ratio = float(head_args[3]) if len(head_args) > 3 else 2.0
            preprocessing_config = model_spec.get("preprocessing") or ANIMALVITPOSE_PREPROCESSING
            augmentation_config = dict(ANIMALVITPOSE_AUGMENTATION)
            augmentation_config.update({
                key: values[key]
                for key in augmentation_config
                if key in values
            })
            validation_config = dict(ANIMALVITPOSE_VALIDATION)
            validation_config.update({
                key: values[key]
                for key in validation_config
                if key in values
            })
            if len(simcc_input_size) != 2 or simcc_input_size[0] != simcc_input_size[1]:
                raise ValueError("AnimalViTPose currently requires a square SimCC input size")
            if config.multi_scale:
                raise ValueError("multi_scale is incompatible with the fixed-size SimCCHead")
            if rank == 0:
                emitter.emit(TrainingEvent(
                    event="model_variant",
                    metrics={
                        "input_width": float(simcc_input_size[0]),
                        "input_height": float(simcc_input_size[1]),
                    },
                    message=f"AnimalViTPose {variant_name}; input_size={simcc_input_size}.",
                ))
                emitter.emit(TrainingEvent(
                    event="augmentation",
                    message=(
                        "Using the MMPose top-down pipeline (instance crop, horizontal flip, "
                        "half-body when keypoint groups are supplied, and random bbox scale/shift/rotation). "
                        "YOLO Mosaic, MixUp and HSV settings do not apply to AnimalViTPose."
                    ),
                ))
            from animalposetracker.training.simcc import SimCCKLLoss
            from animalposetracker.evaluation.topdown import SimCCPoseValidator
            from animalposetracker.data.topdown import build_topdown_dataloaders

            train_loader, validation_loader, metadata = build_topdown_dataloaders(
                config.data,
                input_size=simcc_input_size,
                batch_size=config.batch_size // world_size,
                validation_batch_size=(
                    config.validation_batch_size // world_size
                    if config.validation_batch_size is not None else None
                ),
                num_workers=config.num_workers,
                seed=config.seed,
                include_validation=config.validation_enabled,
                cache=config.cache,
                fraction=config.fraction,
                sigma=float(values.get("label_sigma", 6.0)),
                split_ratio=simcc_split_ratio,
                augmentation_config=augmentation_config,
                preprocessing_config=preprocessing_config,
            )
            config.train_dataset_size = metadata["train_instances"]
            set_simcc_keypoint_count(
                model_spec,
                metadata["kpt_shape"][0],
                metadata["kpt_shape"],
            )
            model_spec["nc"] = 1
            model = build_model(model_spec, scale=scale)
            head = model.model[-1]
            if not isinstance(head, SimCCHead):
                raise TypeError(f"Expected SimCCHead, built {type(head).__name__}")
            if list(model.kpt_shape) != list(metadata["kpt_shape"]):
                raise ValueError(
                    f"Model keypoint shape {list(model.kpt_shape)} does not match the dataset "
                    f"shape {metadata['kpt_shape']}"
                )
            criterion = SimCCKLLoss(
                metadata["kpt_shape"][0],
                beta=float(values.get("kl_beta", 1.0)),
            )
            validator = (
                SimCCPoseValidator(
                    input_size=simcc_input_size,
                    split_ratio=simcc_split_ratio,
                    flip_indices=metadata["flip_idx"],
                    kpt_oks_sigmas=metadata["kpt_oks_sigmas"],
                    validation_dataset=validation_loader.dataset,
                    # Validation-only flip TTA, off by default so the reported metrics match what
                    # the shipped decoder (decode_simcc) actually produces. Opt in with
                    # `validation.flip_tta: true` in the model preset.
                    flip_tta=bool(validation_config.get("flip_tta", False)),
                    coco_max_detections=int(values.get(
                        "coco_max_detections",
                        validation_config.get("coco_max_detections", 20),
                    )),
                    oks_nms_threshold=float(validation_config.get("oks_nms_threshold", 0.9)),
                    keypoint_score_threshold=float(
                        validation_config.get("keypoint_score_threshold", 0.2)
                    ),
                    pck_threshold=float(values.get(
                        "pck_threshold", KEYPOINT_METRICS.get("pck_threshold", 0.05)
                    )),
                    auc_norm_factor=float(values.get(
                        "auc_norm_factor", KEYPOINT_METRICS.get("auc_norm_factor", 30.0)
                    )),
                    auc_thresholds=int(values.get(
                        "auc_thresholds", KEYPOINT_METRICS.get("auc_thresholds", 20)
                    )),
                    oks_area_mode=config.oks_area_mode,
                    ap_metric=(
                        config.best_metric
                        if config.best_metric in {"COCO/AP", "YOLO/AP"}
                        else "COCO/AP"
                    ),
                    max_detections=config.max_detections,
                )
                if validation_loader is not None else None
            )
        else:
            train_loader, validation_loader, metadata = build_pose_dataloaders(
                config.data,
                image_size=config.image_size,
                batch_size=config.batch_size // world_size,
                validation_batch_size=(
                    config.validation_batch_size // world_size
                    if config.validation_batch_size is not None else None
                ),
                num_workers=config.num_workers,
                seed=config.seed,
                include_validation=config.validation_enabled,
                cache=config.cache,
                fraction=config.fraction,
                augmentation={
                    "hsv_h": config.hsv_h,
                    "hsv_s": config.hsv_s,
                    "hsv_v": config.hsv_v,
                    "degrees": config.degrees,
                    "translate": config.translate,
                    "scale": config.scale,
                    "shear": config.shear,
                    "perspective": config.perspective,
                    "flipud": config.flipud,
                    "fliplr": config.fliplr,
                    "bgr": config.bgr,
                    "mosaic": config.mosaic,
                    "mixup": config.mixup,
                },
                epochs=config.epochs,
                close_mosaic=config.close_mosaic,
                single_cls=config.single_cls,
                device=config.device,
            )
            model = build_model(model_spec, scale=scale)
            head = model.model[-1]
            if not isinstance(head, YOLOPoseHead):
                raise NotImplementedError(
                    f"No AnimalPoseTracker loss is registered for {type(head).__name__}"
                )
            config.train_dataset_size = metadata["train_images"]
            if int(model.nc) != int(metadata["num_classes"]):
                raise ValueError(
                    f"Model has {model.nc} classes but the dataset defines {metadata['num_classes']}"
                )
            criterion = PoseDetectionLoss(
                head,
                image_size=config.image_size,
                kpt_oks_sigmas=metadata["kpt_oks_sigmas"],
                box_weight=config.box_loss_weight,
                class_weight=config.class_loss_weight,
                distribution_weight=config.distribution_loss_weight,
                keypoint_weight=config.keypoint_loss_weight,
                visibility_weight=config.visibility_loss_weight,
            )
            validator = (
                PoseDetectionValidator(
                    criterion,
                    confidence_threshold=config.validation_confidence,
                    iou_threshold=config.validation_iou,
                    max_detections=config.max_detections,
                    agnostic_nms=config.agnostic_nms,
                    plots=config.plots and rank == 0,
                    output_dir=str(config.output_dir),
                    class_names=metadata["class_names"],
                    validation_dataset=(
                        validation_loader.dataset if validation_loader is not None else None
                    ),
                    keypoint_pck_threshold=float(KEYPOINT_METRICS.get("pck_threshold", 0.05)),
                    keypoint_auc_norm_factor=float(KEYPOINT_METRICS.get("auc_norm_factor", 30.0)),
                    keypoint_auc_thresholds=int(KEYPOINT_METRICS.get("auc_thresholds", 20)),
                    coco_max_detections=config.max_detections,
                    oks_area_mode=config.oks_area_mode,
                    ap_metric=(
                        config.best_metric
                        if config.best_metric in {"COCO/AP", "YOLO/AP"}
                        else "COCO/AP"
                    ),
                )
                if validation_loader is not None else None
            )

        if list(model.kpt_shape) != list(metadata["kpt_shape"]):
            raise ValueError(
                f"Model keypoint shape {list(model.kpt_shape)} does not match the dataset "
                f"shape {metadata['kpt_shape']}"
            )

        if config.pretrained_weights is not None and not args.validate_only:
            pretrained_path = _resolve_project_pretrained_path(
                config.pretrained_weights,
                config.project_dir,
            )
            if not pretrained_path.is_file():
                raise FileNotFoundError(f"Pretrained weights do not exist: {pretrained_path}")
            report = load_model_weights(pretrained_path, model, strict=False)
            emitter.emit(TrainingEvent(
                event="pretrained",
                metrics={
                    "loaded_tensors": float(report["loaded_tensors"]),
                    "shape_mismatches": float(len(report["shape_mismatches"])),
                    "unexpected_tensors": float(len(report["unexpected_keys"])),
                },
                message=(
                    f"Loaded {report['loaded_tensors']} tensors; "
                    f"skipped {len(report['shape_mismatches'])} incompatible shapes and "
                    f"{len(report['unexpected_keys'])} checkpoint-only tensors."
                ),
            ))

        seed_everything(rank_seed, config.deterministic)

        trainer = Trainer(model, config, events=emitter)
        if args.validate_only:
            if validation_loader is None or validator is None:
                raise ValueError("Evaluation requires a validation split in dataset.yaml")
            weights_path = _load_evaluation_weights(args.weights, trainer, emitter)
            evaluation_model = trainer.ema.model if trainer.ema.enabled else trainer.raw_model
            evaluation_model.eval()
            with trainer.torch.inference_mode():
                metrics = validator(evaluation_model, validation_loader, trainer.device)
            metrics = {str(key): float(value) for key, value in metrics.items()}
            result_path = config.output_dir / "metrics.json"
            result_path.write_text(
                json.dumps(
                    {
                        "weights": str(weights_path),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            emitter.emit(TrainingEvent(
                event="evaluation",
                metrics=metrics,
                message=f"Evaluation complete; metrics saved to {result_path}",
            ))
            return 0
        emitter.emit(TrainingEvent(
            event="optimizer",
            metrics={
                "learning_rate": float(config.resolved_learning_rate),
                "momentum": float(config.resolved_momentum),
                "weight_decay": float(config.resolved_weight_decay),
            },
            message=f"{config.resolved_optimizer}; gradient accumulation={config.gradient_accumulation_steps}",
        ))

        stop_marker = config.project_dir / ".animalposetracker_training_stop"
        stop_watcher_done = threading.Event()

        def watch_stop_marker():
            while not stop_watcher_done.wait(0.25):
                if stop_marker.is_file():
                    trainer.request_stop()
                    return

        stop_watcher = threading.Thread(target=watch_stop_marker, daemon=True)
        stop_watcher.start()

        def request_stop(_signum, _frame):
            trainer.request_stop()

        signal.signal(signal.SIGINT, request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, request_stop)

        try:
            result = trainer.fit(
                train_loader,
                criterion,
                validation_loader=validation_loader,
                validator=validator,
            )
            if distributed_initialized:
                import torch

                torch.distributed.barrier()
        finally:
            stop_watcher_done.set()
            stop_watcher.join(timeout=1.0)
            if rank == 0 and stop_marker.exists():
                stop_marker.unlink()
        # Early stopping is a normal, successful end of training, so it exits 0; only an
        # external stop request (SIGINT/SIGTERM or the stop marker) reports 130. Returning
        # 130 for an exhausted patience budget used to abort `set -e` pipelines right after
        # `train`, so the val/export steps chained behind it never ran.
        if result.get("early_stopped"):
            return 0
        return 0 if not result["stopped"] else 130
    except Exception as exc:
        if trainer is None:
            emitter.emit(TrainingEvent(event="failed", message=str(exc)))
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if event_stream is not None:
            event_stream.close()
        if distributed_initialized:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
