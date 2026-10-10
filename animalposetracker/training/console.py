"""Human-readable console rendering for training progress events.

The machine-readable contract is unchanged: ``training.jsonl`` always
receives JSON-lines. This renderer only affects what a person sees on an
interactive terminal.
"""

import sys
from typing import Optional, Sequence, TextIO, Tuple

from .events import TrainingEvent

_LOSS_KEYS = ("loss", "box", "pose", "kobj", "class", "dfl")
_VAL_KEYS = ("loss", "coco/AP", "coco/AP50", "coco/AR", "PCK")


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


class PrettyTrainingRenderer:
    """Render :class:`TrainingEvent` objects as compact readable lines."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self._batch_open = False

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _line(self, text: str) -> None:
        if self._batch_open:
            self._write("\n")
            self._batch_open = False
        self._write(text + "\n")

    @staticmethod
    def _metrics(metrics, keys) -> str:
        parts = []
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                parts.append(f"{key}={_fmt(float(value))}")
        return " ".join(parts)

    def emit(self, event: TrainingEvent) -> None:
        metrics = event.metrics or {}
        name = event.event

        if name == "batch":
            text = (
                f"epoch {event.epoch}/{event.epochs} "
                f"step {event.step}/{event.steps} "
                + self._metrics(metrics, _LOSS_KEYS)
            )
            self._write("\r" + text.ljust(110))
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
            self._line(
                f"[val] epoch {event.epoch}/{event.epochs} "
                + self._metrics(metrics, _VAL_KEYS)
            )
            return

        if name == "epoch_end":
            train_loss = metrics.get("train_loss")
            val_loss = metrics.get("val_loss")
            ap = metrics.get("val_coco/AP", metrics.get("coco/AP"))
            parts = [f"[epoch {event.epoch}/{event.epochs}]"]
            if train_loss is not None:
                parts.append(f"train_loss={_fmt(float(train_loss))}")
            if val_loss is not None:
                parts.append(f"val_loss={_fmt(float(val_loss))}")
            if ap is not None:
                parts.append(f"AP={_fmt(float(ap))}")
            self._line(" ".join(parts))
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
