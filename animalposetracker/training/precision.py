"""Pick the fastest mixed-precision mode whose forward pass stays numerically finite.

fp16 saturates at 65504: once an activation crosses it the forward returns inf and the
loss turns nan, which surfaces much later as a dead run. bf16 shares fp32's exponent
range and cannot saturate that way, but it needs Ampere or newer. The probe here runs a
forward on synthetic input only — no dataset, no trained weights — so it answers "can
this model structure run at this precision on this card", not "is the model converging".
"""

from __future__ import annotations

from typing import Any

PRECISIONS = ("fp16", "bf16", "fp32")


def _iter_tensors(value: Any):
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def resolve_precision(model, image_size: int, requested: Any = True) -> str:
    """Return ``"fp16"``, ``"bf16"`` or ``"fp32"`` for this model on this device.

    ``requested`` follows the ``amp`` setting: ``False`` pins fp32, a precision name
    forces that mode, and ``True`` walks fp16 → bf16 → fp32 until a forward stays finite.
    """
    import torch

    if requested is False:
        return "fp32"
    if isinstance(requested, str):
        name = requested.strip().lower()
        if name in PRECISIONS:
            return name
    if not torch.cuda.is_available():
        return "fp32"

    candidates = ["fp16"]
    if torch.cuda.is_bf16_supported():
        candidates.append("bf16")

    device = next(model.parameters()).device
    if device.type != "cuda":
        if not torch.cuda.is_available():
            return "fp32"
        model.to("cuda")
        device = next(model.parameters()).device
    sample = torch.randn(2, 3, image_size, image_size, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for name in candidates:
                dtype = torch.float16 if name == "fp16" else torch.bfloat16
                try:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        output = model(sample)
                except RuntimeError:  # unsupported op for this dtype
                    continue
                if all(torch.isfinite(tensor).all() for tensor in _iter_tensors(output)):
                    return name
        return "fp32"
    finally:
        model.train(was_training)


def amp_dtype(precision: str):
    """Map a precision name to the autocast dtype (``None`` disables autocast)."""
    import torch

    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


__all__ = ["PRECISIONS", "amp_dtype", "resolve_precision"]
