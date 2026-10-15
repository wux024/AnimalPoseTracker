"""Probe the largest batch size that fits inside a fraction of GPU memory.

Mirrors ultralytics' auto-batch trick: run real forward/backward steps at a
modest batch size, measure the peak reserved memory, then walk a candidate
list of larger sizes until the measured peak crosses the target (60% of VRAM
by default). The winner is returned as-is; candidates are multiples of 8.

The synthetic loss used while probing does not skew the memory profile:
activations, gradients and optimizer state dominate, and those are all real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

AUTO = "auto"
DEFAULT_CANDIDATES: Sequence[int] = (128, 96, 64, 48, 32, 16)


def parse_batch_size(value: Any) -> Union[int, str]:
    """Accept ``auto`` / ``-1`` (probe at runtime) or a concrete integer batch size."""
    if isinstance(value, bool):
        raise ValueError(f"batch size must be an integer or 'auto', got: {value!r}")
    if isinstance(value, int):
        return AUTO if value < 0 else value
    text = str(value).strip().lower()
    if text in {"auto", "-1"}:
        return AUTO
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"batch size must be an integer or 'auto', got: {value!r}") from exc


def _resolve_probe_device(device: str):
    import torch

    dev = str(device or "auto").strip().lower()
    if dev in {"", "auto"}:
        dev = "cuda:0"
    elif dev.isdigit():
        dev = f"cuda:{dev}"
    if not dev.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("--batch-size auto requires a CUDA device")
    return torch.device(dev)


def probe_batch_size(
    data_yaml: Union[str, Path],
    model_yaml: Union[str, Path],
    model_scale: str,
    image_size: int,
    device: str = "auto",
    fraction: float = 0.6,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    warmup_steps: int = 3,
    fraction_of_dataset: float = 0.3,
) -> int:
    """Return the largest candidate batch size whose peak stays inside ``fraction`` of VRAM.

    Runs ``warmup_steps`` real forward/backward/optimizer-step passes per candidate
    (on a small slice of the real training data) and compares
    ``torch.cuda.max_memory_reserved`` against ``fraction`` of the total memory.
    A CUDA out-of-memory error at any size simply moves the probe to the next
    smaller candidate.
    """
    import torch
    from ..data.pose import build_pose_dataloaders
    from ..nn import build_model

    resolved = _resolve_probe_device(device)
    total_gb = torch.cuda.get_device_properties(resolved).total_memory / 2 ** 30
    target_gb = total_gb * fraction

    model = build_model(model_yaml, scale=model_scale).to(resolved).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for batch_size in candidates:
        try:
            train_loader, _, _ = build_pose_dataloaders(
                data_yaml,
                image_size,
                batch_size,
                num_workers=2,
                include_validation=False,
                fraction=fraction_of_dataset,
                epochs=1,
                device=str(resolved),
            )
            iterator = iter(train_loader)
            steps = max(1, min(warmup_steps, len(train_loader)))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(steps):
                batch = next(iterator)
                images = batch["images"].to(resolved).float().div_(255.0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=True):
                    outputs = model(images)
                tensors = (
                    list(outputs.values())
                    if isinstance(outputs, dict)
                    else list(outputs)
                    if isinstance(outputs, (list, tuple))
                    else [outputs]
                )
                total = sum(t.float().sum() for t in tensors if torch.is_tensor(t))
                scaler.scale(total).backward()
                scaler.step(optimizer)
                scaler.update()
            peak_gb = torch.cuda.max_memory_reserved() / 2 ** 30
            if peak_gb <= target_gb:
                return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return candidates[-1]
