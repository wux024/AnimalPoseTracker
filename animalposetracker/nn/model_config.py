"""Model-head configuration helpers shared by project and training workflows."""

from typing import Any, Dict, Optional, Tuple


def head_config(model_spec: Dict[str, Any]) -> Tuple[Optional[str], Optional[list]]:
    entries = model_spec.get("head") or []
    if not isinstance(entries, list) or not entries:
        return None, None
    entry = entries[-1]
    if not isinstance(entry, (list, tuple)) or len(entry) < 4:
        return None, None
    return str(entry[2]), entry


def set_simcc_keypoint_count(model_spec: Dict[str, Any], count: int, shape) -> None:
    model_spec["nc"] = 1
    model_spec["kpt_shape"] = list(shape)
    _module_name, entry = head_config(model_spec)
    if entry is None:
        raise ValueError("SimCC model configuration has no head layer")
    args = entry[3]
    if not isinstance(args, list) or not args:
        raise ValueError("SimCCHead configuration must use a positional argument list")
    args[0] = int(count)


__all__ = ["head_config", "set_simcc_keypoint_count"]
