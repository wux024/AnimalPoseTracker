"""Terminal presentation shared by every command's startup banner.

``create``, ``train``, ``val``, ``predict``, ``export`` and ``infer`` all open
with the same two things: a one-line runtime summary and an aligned
``key  value`` block. Those two helpers live here, at the package root, so that
the non-training commands do not have to import the training package to print a
banner -- ``console.py`` used to sit inside ``training/`` and every other
subsystem reached back into it for roughly thirty lines of string formatting.

Torch is imported lazily inside :func:`environment_line` and only used to enrich
the line when it happens to be installed, so this module stays importable in a
deployment environment that has no training stack. Training-specific rendering
(progress tables, the ``tqdm`` bridge) remains in
:mod:`animalposetracker.training.console`.
"""

from typing import Sequence, Tuple

__all__ = ["environment_line", "format_summary_block", "describe_pretrained"]


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
    """Aligned ``key  value`` block used by the create/train/predict/export/infer banners."""
    width = max(len(key) for key, _ in rows)
    return "\n".join(f"{indent}{key.ljust(width)}  {value}" for key, value in rows)


def describe_pretrained(value) -> str:
    """Render a project's ``pretrained`` setting for the create banner."""
    if value is True:
        return "auto (resolved from the project pretrained/ directory at train time)"
    if value in (False, None):
        return "disabled (train from scratch)"
    return str(value)
