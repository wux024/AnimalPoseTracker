"""Human-readable console rendering for training progress events.

The machine-readable contract is unchanged: ``training.jsonl`` always
receives JSON-lines. This renderer only affects what a person sees on an
interactive terminal.

The startup-banner helpers shared with the other commands
(``environment_line``, ``format_summary_block``, ``describe_pretrained``) live in
:mod:`animalposetracker.console`; this module holds only the training-specific
progress rendering.
"""

import sys
from typing import NamedTuple, Optional, TextIO, Tuple

from tqdm import tqdm

from .events import TrainingEvent

# The keypoint metrics both heads report: PCK/AUC/EPE come straight from the
# datasets, and the AP block is picked from whichever namespace the validator
# filled in (see `_select_ap_namespace`).
_KEYPOINT_VALIDATION_COLUMNS = (
    ("coco/AP", "AP"),
    ("coco/AP50", "AP50"),
    ("coco/AP75", "AP75"),
    ("ultralytics/AP", "UltraAP"),
    ("ultralytics/AP50", "UltraAP50"),
    ("ultralytics/AP75", "UltraAP75"),
    ("ultralytics/P", "P"),
    ("ultralytics/R", "R"),
    ("coco/AR", "AR"),
    ("PCK", "PCK"),
    ("AUC", "AUC"),
    ("EPE", "EPE"),
)


class LossColumns(NamedTuple):
    """The loss columns a head reports, as ``(metric key, header label)`` pairs.

    A layout is a translation table, not a filter. The keys are exactly the ones
    the criterion hands to ``engine._loss_components`` (``YOLOPoseHead`` reports
    ``loss/box/pose/kobj/class/dfl``, ``SimCCHead`` reports ``loss`` plus the two
    SimCC axes), so a column that renders as ``-`` means the head structurally
    never reports that key -- the row would print an empty slot every single step.
    """

    train: Tuple[Tuple[str, str], ...]
    validation: Tuple[Tuple[str, str], ...]


YOLO_POSE_COLUMNS = LossColumns(
    train=(
        ("loss", "loss"),
        ("box", "box_loss"),
        ("pose", "pose_loss"),
        ("kobj", "kobj_loss"),
        ("class", "cls_loss"),
        ("dfl", "dfl_loss"),
    ),
    validation=(("loss", "Loss"), *_KEYPOINT_VALIDATION_COLUMNS),
)

SIMCC_COLUMNS = LossColumns(
    # The KL head's `loss` is a batch sum (MMPose divides by the keypoint count, not
    # by the batch), so it doubles when batch_size does and cannot be read as a
    # training curve. Its per-image average is the comparable figure, and it is the
    # only loss column worth a permanent slot; the x/y split and the raw sum stay in
    # training.jsonl for anyone debugging which axis drifted.
    train=(("loss_simcc_per_sample", "loss"),),
    # The SimCC validator scores AP/PCK/AUC/EPE but computes no validation loss, so
    # this table has no Loss column to leave empty.
    validation=_KEYPOINT_VALIDATION_COLUMNS,
)


def loss_columns(head_name: Optional[str]) -> LossColumns:
    """Pick the column layout for a detection head, defaulting to the YOLO-Pose one."""
    return SIMCC_COLUMNS if head_name == "SimCCHead" else YOLO_POSE_COLUMNS


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


def _select_ap_namespace(metrics, prefix: str) -> str:
    """Which AP block the validator filled in; the other is absent, not zero."""
    return "ultralytics" if f"{prefix}ultralytics/AP" in metrics else "coco"


def _validation_table(metrics, columns, prefix: str = "") -> Tuple[str, str]:
    selected_ap_prefix = _select_ap_namespace(metrics, prefix)
    selected = []
    for key, label in columns:
        if key.startswith("coco/"):
            if selected_ap_prefix == "coco":
                selected.append((key, label))
        elif key.startswith("ultralytics/"):
            if selected_ap_prefix == "ultralytics":
                selected.append((key, label))
        else:
            selected.append((key, label))
    selected = tuple(selected)
    headers = ("Images", "Instances", *(label for _, label in selected))
    header = f"{'Class':>22s}" + "".join(f"{label:>11s}" for label in headers)
    image_count = metrics.get(f"{prefix}images")
    instance_count = metrics.get(f"{prefix}instances")
    values = [
        "all",
        str(int(image_count)) if image_count is not None else "-",
        str(int(instance_count)) if instance_count is not None else "-",
    ]
    for key, _ in selected:
        value = metrics.get(f"{prefix}{key}")
        values.append(_fmt(float(value)) if value is not None else "-")
    row = f"{values[0]:>22s}" + "".join(f"{value:>11s}" for value in values[1:])
    return header, row


class PrettyTrainingRenderer:
    """Render training batches with a terminal progress bar like Ultralytics."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        columns: LossColumns = YOLO_POSE_COLUMNS,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        # Which loss columns the progress row carries. The CLI swaps this once it knows
        # which head the model holds (`loss_columns`), defaulting to the YOLO-Pose layout.
        self.columns = columns
        self._header_epoch = None
        self._progress = None

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    def _line(self, text: str) -> None:
        self._close_progress()
        self.stream.write(text + "\n")
        self.stream.flush()

    def emit(self, event: TrainingEvent) -> None:
        metrics = event.metrics or {}
        name = event.event

        if name == "batch":
            train_columns = self.columns.train
            if self._header_epoch != event.epoch:
                self._close_progress()
                headers = (
                    "Epoch", "GPU_mem", *(label for _, label in train_columns), "Instances", "Size"
                )
                self._line("".join(f"{label:>11s}" for label in headers))
                self._header_epoch = event.epoch
            total_steps = max(int(event.steps or 0), 1)
            if self._progress is None:
                self._progress = tqdm(
                    total=total_steps,
                    file=self.stream,
                    dynamic_ncols=True,
                    leave=True,
                    unit="it",
                    mininterval=0,
                    disable=False,
                )
            description_values = [
                f"{event.epoch}/{event.epochs}",
                f"{float(metrics.get('gpu_mem', 0.0)):.3g}G",
            ]
            description_values.extend(
                _fmt(float(metrics[key])) if key in metrics else "-"
                for key, _ in train_columns
            )
            description_values.extend((
                str(int(metrics.get("instances", 0))),
                str(int(metrics.get("size", 0))),
            ))
            description = "".join(f"{value:>11s}" for value in description_values)
            self._progress.set_description(description, refresh=False)
            step = min(max(int(event.step or 0), 0), total_steps)
            self._progress.update(max(step - self._progress.n, 0))
            if self._progress.n >= total_steps:
                self._close_progress()
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
            header, row = _validation_table(metrics, self.columns.validation)
            self._line(header)
            self._line(row)
            speed = _speed_text(metrics)
            if speed:
                self._line(speed)
            return

        if name == "evaluation":
            header, row = _validation_table(metrics, self.columns.validation)
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
            header, row = _validation_table(metrics, self.columns.validation, prefix="val_")
            self._line(header)
            self._line(row)
            return

        if name in {"started", "finished", "stopped", "plot", "profile"}:
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
