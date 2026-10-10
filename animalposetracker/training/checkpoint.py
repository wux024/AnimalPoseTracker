"""Framework-owned checkpoint persistence."""

import os
import importlib
import random
import re
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


CHECKPOINT_FORMAT_VERSION = 1


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for model training. Install AnimalPoseTracker's training extra."
        ) from exc
    return torch


def read_checkpoint_config(path) -> Optional[Dict[str, Any]]:
    """Read saved training arguments before building a resumed training run."""
    torch = _torch()
    try:
        payload = torch.load(str(path), map_location="meta", weights_only=False)
    except TypeError:
        try:
            payload = torch.load(str(path), map_location="meta")
        except (TypeError, RuntimeError, ValueError):
            payload = torch.load(str(path), map_location="cpu")
    except (RuntimeError, ValueError):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return None
    config = payload.get("config")
    return dict(config) if isinstance(config, Mapping) else None


def resolve_checkpoint_path(source, cache_dir=None) -> Path:
    """Resolve a local checkpoint or download a remote pretrained file into the user cache."""
    source_text = str(source)
    if not source_text.lower().startswith(("http://", "https://")):
        return Path(source_text).expanduser().resolve()

    filename = Path(urlparse(source_text).path).name
    if not filename:
        raise ValueError(f"Pretrained checkpoint URL has no filename: {source_text}")
    cache_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path.home() / ".cache" / "animalposetracker" / "pretrained"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    lock_path = cache_dir / f".{filename}.lock"
    deadline = time.monotonic() + 1800.0

    while True:
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 1800.0:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for pretrained checkpoint download: {source_text}")
            time.sleep(0.25)

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.download")
    try:
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        with urlopen(source_text, timeout=60) as response, temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded pretrained checkpoint is empty: {source_text}")
        os.replace(temporary, destination)
        return destination
    except Exception as exc:
        raise RuntimeError(f"Could not download pretrained checkpoint from {source_text}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def _rng_state() -> Dict[str, Any]:
    torch = _torch()
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    torch = _torch()
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _legacy_pose_globals():
    """Map known legacy checkpoint classes to this project's local module classes."""
    from animalposetracker.nn.attention import ChannelAttention
    from animalposetracker.nn.builder import GraphSequential
    from animalposetracker.nn.head import YOLOPoseHead
    from animalposetracker.nn.modules import (
        CSPNeXtBlock,
        CSPNeXtBottleneck,
        Concat,
        Conv,
        DFL,
        DWConv,
        SPPF,
        STEM,
    )

    local_classes = {
        "ultralytics.nn.tasks.PoseModel": GraphSequential,
        "ultralytics.nn.modules.block.STEM": STEM,
        "ultralytics.nn.modules.conv.Conv": Conv,
        "ultralytics.nn.modules.block.CSPNeXtBlock": CSPNeXtBlock,
        "ultralytics.nn.modules.block.CSPNeXtBottleneck": CSPNeXtBottleneck,
        "ultralytics.nn.modules.conv.DWConv": DWConv,
        "ultralytics.nn.modules.conv.ChannelAttention": ChannelAttention,
        "ultralytics.nn.modules.block.SPPF": SPPF,
        "ultralytics.nn.modules.conv.Concat": Concat,
        "ultralytics.nn.modules.head.Pose": YOLOPoseHead,
        "ultralytics.nn.modules.block.DFL": DFL,
    }
    aliases = []
    for qualified_name, local_class in local_classes.items():
        module_name, _, class_name = qualified_name.rpartition(".")
        aliases.append(type(class_name, (local_class,), {"__module__": module_name}))
    return aliases


def _numpy_checkpoint_globals():
    """Allow only NumPy's array/dtype primitives used by the saved RNG state."""
    try:
        import numpy as np
    except ImportError:
        return []

    core = getattr(np, "_core", None)
    if core is None:
        core = np.core
    multiarray = core.multiarray
    return [
        multiarray._reconstruct,
        multiarray.scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]


def _load_tensor_checkpoint(path, torch, map_location="cpu"):
    """Safely read tensor checkpoints, including known legacy pose model objects.

    Legacy class names are redirected to local classes (or allowlisted native
    PyTorch modules). No package containing a serialized training model is
    imported while reading the checkpoint.
    """
    allowed = _legacy_pose_globals() + _numpy_checkpoint_globals() + [set]
    seen = set()
    for _ in range(64):
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(str(path), map_location=map_location, weights_only=True)
        except Exception as exc:
            match = re.search(r"Unsupported global: GLOBAL ([A-Za-z0-9_.]+)", str(exc))
            if match is None:
                raise
            qualified_name = match.group(1)
            if qualified_name in seen:
                raise ValueError(f"Checkpoint repeats an unsupported global: {qualified_name}") from exc
            seen.add(qualified_name)
            if qualified_name.startswith("torch.nn."):
                module_name, _, class_name = qualified_name.rpartition(".")
                module = importlib.import_module(module_name)
                allowed.append(getattr(module, class_name))
                continue
            raise ValueError(
                f"Checkpoint contains an unsupported serialized class: {qualified_name}. "
                "This checkpoint architecture is not registered for safe local conversion."
            ) from exc
    raise ValueError("Checkpoint contains too many serialized classes for safe local conversion")


def _extract_model_state(payload, torch) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Weights file does not contain a state dictionary or model checkpoint")
    for key in ("ema_state_dict", "ema", "model_state_dict", "state_dict", "model", "weights"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
        if isinstance(candidate, torch.nn.Module):
            return candidate.state_dict()
    return payload


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch=0,
    global_step=0,
    best_metric=None,
    epochs_without_improvement=0,
    config=None,
    ema_model=None,
    ema_updates=None,
) -> None:
    """Atomically save training state so an interrupted write cannot replace a checkpoint."""
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "ema_updates": int(ema_updates) if ema_updates is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "rng_state": _rng_state(),
    }

    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    restore_rng=True,
    ema_model=None,
    prefer_ema_for_model=False,
) -> Dict[str, Any]:
    """Restore a framework checkpoint and return its resume metadata."""
    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    try:
        payload = torch.load(str(source), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        payload = torch.load(str(source), map_location=map_location)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(
            f"Not an AnimalPoseTracker training checkpoint: {source}. "
            "Use load_model_weights() to initialize from a weights-only file."
        )

    model_state = payload.get("ema_state_dict") if prefer_ema_for_model else None
    if not isinstance(model_state, Mapping):
        model_state = payload["model_state_dict"]
    model.load_state_dict(model_state)
    if ema_model is not None:
        ema_model.load_state_dict(payload.get("ema_state_dict") or payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        _restore_rng_state(payload.get("rng_state"))
    ema_updates = payload.get("ema_updates")
    if ema_updates is None:
        ema_updates = payload.get("global_step", 0)
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "ema_updates": int(ema_updates),
        "best_metric": payload.get("best_metric"),
        "epochs_without_improvement": int(payload.get("epochs_without_improvement", 0)),
        "config": payload.get("config"),
        "format_version": payload.get("format_version", 0),
    }


def load_model_weights(path, model, map_location="cpu", strict=True):
    """Load compatible tensors from a plain or safely mapped legacy pose checkpoint."""
    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {source}")
    payload = _load_tensor_checkpoint(source, torch, map_location=map_location)
    state = _extract_model_state(payload, torch)
    target = model.state_dict()
    compatible = {}
    shape_mismatches = []
    unexpected = []
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        normalized_name = name[7:] if name.startswith("module.") else name
        candidate_names = [normalized_name]
        # MMPose checkpoints store the graph under ``backbone`` and ``head`` while
        # AnimalPoseTracker's YAML graph stores those modules at layer 0 and 1.
        if normalized_name.startswith("backbone."):
            candidate_names.append("model.0." + normalized_name[len("backbone."):])
        elif normalized_name.startswith("head."):
            candidate_names.append("model.1." + normalized_name[len("head."):])
        elif normalized_name.startswith("model.backbone."):
            candidate_names.append("model.0." + normalized_name[len("model.backbone."):])
        elif normalized_name.startswith("model.head."):
            candidate_names.append("model.1." + normalized_name[len("model.head."):])
        elif normalized_name.startswith(("patch_embed.", "layers.", "pos_embed", "ln1.")):
            # MMPose's MAE initializer loads a backbone-only state dict without its
            # ``backbone.`` wrapper, unlike a full MMPose training checkpoint.
            candidate_names.append("model.0." + normalized_name)

        target_name = next((candidate for candidate in candidate_names if candidate in target), None)
        if target_name is None:
            unexpected.append(normalized_name)
        else:
            if (
                target_name.endswith("pos_embed")
                and tensor.ndim == 3
                and target[target_name].ndim == 3
                and tensor.shape[0] == target[target_name].shape[0]
                and tensor.shape[2] == target[target_name].shape[2]
                and tensor.shape[1] != target[target_name].shape[1]
            ):
                try:
                    from animalposetracker.nn.transformer import resize_pos_embed

                    owner_name = target_name.rsplit(".", 1)[0]
                    owner = dict(model.named_modules()).get(owner_name)
                    extra_tokens = int(getattr(owner, "num_extra_tokens", 0))
                    tensor = resize_pos_embed(
                        tensor,
                        target[target_name].shape[1],
                        extra_tokens,
                    )
                except (ImportError, TypeError, ValueError):
                    pass
            if target[target_name].shape != tensor.shape:
                shape_mismatches.append(
                    (target_name, tuple(tensor.shape), tuple(target[target_name].shape))
                )
                continue
            compatible[target_name] = tensor

    if strict:
        if shape_mismatches or unexpected:
            raise ValueError(
                f"Weights do not exactly match the model: {len(shape_mismatches)} shape mismatches, "
                f"{len(unexpected)} unexpected tensors"
            )
        return model.load_state_dict(compatible, strict=True)
    if not compatible:
        raise ValueError(f"No compatible model tensors were found in {source}")
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
