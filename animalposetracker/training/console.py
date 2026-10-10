"""Human-readable console rendering for training progress events.

The machine-readable contract is unchanged: ``training.jsonl`` always
receives JSON-lines. This renderer only affects what a person sees on an
interactive terminal.
"""

import sys
from typing import Optional, Sequence, TextIO, Tuple

from .events import TrainingEvent

_TRAIN_COLUMNS = (
    ("loss", "loss"),
    ("box", "box_loss"),
    ("pose", "pose_loss"),
    ("kobj", "kobj_loss"),
    ("class", "cls_loss"),
    ("dfl", "dfl_loss"),
)
_VAL_COLUMNS = (
    ("loss", "Loss"),
    ("coco/AP", "AP"),
    ("coco/AP50", "AP50"),
    ("coco/AP75", "AP75"),
    ("coco/AR", "AR"),
    ("PCK", "PCK"),
    ("AUC", "AUC"),
    ("EPE", "EPE"),
)


def environment_line() -> str:
    """One-line runtime summary in the style of Ultralytics banners."""
    import platform

    from animalposetracker import __version__

    parts = [f"AnimalPoseTracker {__version__}", f"Python-{platform.python_version()}"]
    try:
        import torch

        parts.append(f"torch-{torch.__version__}")
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            name = torch.cuda.get_device_name(index)
            memory = torch.cuda.get_device_properties(index).total_memory / (1024 ** 2)
            parts.append(f"CUDA:{index} ({name}, {memory:.0f}MiB)")
        else:
            parts.append("CPU")
    except Exception:
        pass
    return " · ".join(parts)


def format_summary_block(rows: Sequence[Tuple[str, str]], indent: str = "  ") -> str:
    """Aligned ``key  value`` block used by create/train banners."""
    width = max(len(key) for key, _ in rows)
    return "\n".join(f"{indent}{key.ljust(width)}  {value}" for key, value in rows)


def describe_pretrained(value) -> str:
    if value is True:
        return "auto (resolved from the project pretrained/ directory at train time)"
    if value in (False, None):
        return "disabled (train from scratch)"
    return str(value)


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _speed_text(metrics, prefix: str = "") -> str:
    stages = (
        ("preprocess", "preprocess"),
        ("inference", "inference"),
        ("loss", "loss"),
        ("postprocess", "postprocess"),
    )
    values = [
        f"{float(metrics[f'{prefix}speed/{key}_ms']):.1f}ms {label}"
        for key, label in stages
        if f"{prefix}speed/{key}_ms" in metrics
    ]
    return f"Speed: {', '.join(values)} per image" if values else ""


def _validation_table(metrics, prefix: str = "") -> Tuple[str, str]:
    headers = ("Images", "Instances", *(label for _, label in _VAL_COLUMNS))
    header = f"{'Class':>22s}" + "".join(f"{label:>11s}" for label in headers)
    image_count = metrics.get(f"{prefix}images")
    instance_count = metrics.get(f"{prefix}instances")
    values = [
        "all",
        str(int(image_count)) if image_count is not None else "-",
        str(int(instance_count)) if instance_count is not None else "-",
    ]
    for key, _ in _VAL_COLUMNS:
        value = metrics.get(f"{prefix}{key}")
        values.append(_fmt(float(value)) if value is not None else "-")
    row = f"{values[0]:>22s}" + "".join(f"{value:>11s}" for value in values[1:])
    return header, row


def format_progress_bar(current: int, total: int, width: int = 16) -> str:
    """Return a compact fixed-width progress bar for finite workloads."""
    if total <= 0:
        return "░" * width
    progress = min(max(float(current) / total, 0.0), 1.0)
    filled = int(round(progress * width))
    return "█" * filled + "░" * (width - filled)


class PrettyTrainingRenderer:
    """Render :class:`TrainingEvent` objects as compact readable lines."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self._batch_open = False
        self._header_epoch = None

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _line(self, text: str) -> None:
        if self._batch_open:
            self._write("\n")
            self._batch_open = False
        self._write(text + "\n")

    def emit(self, event: TrainingEvent) -> None:
        metrics = event.metrics or {}
        name = event.event

        if name == "batch":
            if self._header_epoch != event.epoch:
                headers = (
                    "Epoch", "GPU_mem", *(label for _, label in _TRAIN_COLUMNS), "Instances", "Size"
                )
                self._line("".join(f"{label:>11s}" for label in headers))
                self._header_epoch = event.epoch
            total_steps = max(int(event.steps or 0), 1)
            progress_bar = format_progress_bar(event.step or 0, total_steps)
            values = [
                f"{event.epoch}/{event.epochs}",
                f"{float(metrics.get('gpu_mem', 0.0)):.3g}G",
            ]
            values.extend(
                _fmt(float(metrics[key])) if key in metrics else "-"
                for key, _ in _TRAIN_COLUMNS
            )
            values.extend((
                str(int(metrics.get("instances", 0))),
                str(int(metrics.get("size", 0))),
            ))
            text = progress_bar + " " + "".join(f"{value:>11s}" for value in values)
            self._write("\r" + text.ljust(145))
            self._batch_open = True
            return

        if name == "pretrained":
            self._line(
                f"[pretrained] loaded {int(metrics.get('loaded_tensors', 0))} tensors"
                f" ({int(metrics.get('shape_mismatches', 0))} incompatible shapes skipped)"
            )
            return

        if name == "optimizer":
            self._line(
                "[optimizer] "
                f"lr={_fmt(float(metrics.get('learning_rate', 0.0)))} "
                f"momentum={_fmt(float(metrics.get('momentum', 0.0)))} "
                f"weight_decay={_fmt(float(metrics.get('weight_decay', 0.0)))}"
                + (f" | {event.message}" if event.message else "")
            )
            return

        if name == "validation":
            return

        if name == "final_validation":
            header, row = _validation_table(metrics)
            self._line(header)
            self._line(row)
            speed = _speed_text(metrics)
            if speed:
                self._line(speed)
            return

        if name == "evaluation":
            header, row = _validation_table(metrics)
            self._line(header)
            self._line(row)
            speed = _speed_text(metrics)
            if speed:
                self._line(speed)
            if event.message:
                self._line(str(event.message))
            return

        if name == "epoch_end":
            train_loss = metrics.get("train_loss")
            parts = [f"[epoch {event.epoch}/{event.epochs}]"]
            if train_loss is not None:
                parts.append(f"train_loss={_fmt(float(train_loss))}")
            self._line(" ".join(parts))
            header, row = _validation_table(metrics, prefix="val_")
            self._line(header)
            self._line(row)
            return

        if name in {"started", "finished", "plot", "profile"}:
            suffix = f" {event.message}" if event.message else ""
            self._line(f"[{name}]{suffix}")
            return

        # Fallback for any other event: compact tag plus message and metrics.
        details = []
        if event.message:
            details.append(str(event.message))
        if metrics:
            details.append(" ".join(f"{key}={_fmt(float(value))}" for key, value in metrics.items()))
        self._line(f"[{name}]" + (" " + " ".join(details) if details else ""))
