"""Model-agnostic PyTorch training loop used by AnimalPoseTracker tasks."""

import math
import os
import random
import threading
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .ema import ModelEMA
from .events import EventEmitter, TrainingEvent


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for model training. Install AnimalPoseTracker's training extra."
        ) from exc
    return torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed the Python, NumPy and PyTorch generators used by a training job."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def initialize_distributed(requested_device: str = "auto") -> Tuple[int, int, int]:
    """Initialize a process group when launched by torchrun and return rank metadata."""
    import os

    torch = _require_torch()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return rank, 1, local_rank
    if not torch.distributed.is_available():
        raise RuntimeError("This PyTorch build does not include torch.distributed")
    if not torch.distributed.is_initialized():
        requested = str(requested_device or "auto").strip().lower()
        wants_cpu = requested in {"cpu", "mps"}
        if torch.cuda.is_available() and not wants_cpu:
            torch.cuda.set_device(local_rank)
            backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        else:
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def _resolve_device(torch, requested: str, local_rank: Optional[int] = None, distributed: bool = False):
    requested = str(requested or "auto").strip().lower()
    if distributed:
        if requested in {"cpu", "mps"}:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda", int(local_rank or 0))
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.isdigit():
        requested = "cuda:" + requested
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available in this PyTorch installation")
    return device


def _make_grad_scaler(torch, enabled: bool):
    scaler_type = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_type is not None:
        try:
            return scaler_type("cuda", enabled=enabled)
        except TypeError:
            return scaler_type(device="cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model, config: TrainingConfig):
    """Build the same optimizer groups and ``auto`` selection used by the pinned pose trainer."""
    torch = _require_torch()
    name = config.optimizer.strip().lower()
    if config.layer_decay_rate is not None:
        if name not in {"adamw", "auto"}:
            raise ValueError("layer_decay_rate is supported with the AnimalViTPose AdamW profile only")
        graph = getattr(model, "model", None)
        backbone = graph[0] if graph is not None and len(graph) else None
        vit_layers = getattr(backbone, "layers", None)
        if vit_layers is None or not len(vit_layers):
            raise TypeError("layer_decay_rate requires a graph model with a VisionTransformer backbone")

        layer_count = len(vit_layers)
        num_max_layer = layer_count + 2
        parameter_groups = {}
        learning_rate = float(config.learning_rate)
        decay = float(config.weight_decay)

        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter_name in {"model.0.cls_token", "model.0.mask_token", "model.0.pos_embed"}
                or parameter_name.startswith("model.0.patch_embed.")
            ):
                layer_id = 0
            elif parameter_name.startswith("model.0.layers."):
                try:
                    layer_id = int(parameter_name.split(".")[3]) + 1
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not determine ViT layer for parameter {parameter_name}") from exc
            else:
                layer_id = num_max_layer - 1

            no_decay = (
                parameter.ndim == 1
                or parameter_name.endswith(".bias")
                or "pos_embed" in parameter_name
            )
            group_name = f"layer_{layer_id}_{'no_decay' if no_decay else 'decay'}"
            if group_name not in parameter_groups:
                lr_scale = config.layer_decay_rate ** (num_max_layer - layer_id - 1)
                parameter_groups[group_name] = {
                    "params": [],
                    "param_names": [],
                    "weight_decay": 0.0 if no_decay else decay,
                    "lr_scale": lr_scale,
                    "lr": learning_rate * lr_scale,
                    "group_name": group_name,
                }
            parameter_groups[group_name]["params"].append(parameter)
            parameter_groups[group_name]["param_names"].append(parameter_name)

        if not parameter_groups:
            raise ValueError("The model has no trainable parameters")
        config.resolved_optimizer = "ADAMW"
        config.resolved_learning_rate = learning_rate
        config.resolved_momentum = float(config.momentum)
        config.resolved_weight_decay = decay
        return torch.optim.AdamW(
            list(parameter_groups.values()),
            lr=learning_rate,
            betas=(float(config.momentum), 0.999),
        )

    parameter_groups = {"bias": [], "decay": [], "norm": []}
    norm_types = tuple(
        module for key, module in torch.nn.__dict__.items()
        if "Norm" in key and isinstance(module, type)
    )
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if "bias" in full_name:
                parameter_groups["bias"].append(parameter)
            elif isinstance(module, norm_types) or "logit_scale" in full_name:
                parameter_groups["norm"].append(parameter)
            else:
                parameter_groups["decay"].append(parameter)
    if not any(parameter_groups.values()):
        raise ValueError("The model has no trainable parameters")

    learning_rate = config.learning_rate
    momentum = config.momentum
    if name == "auto":
        iterations = 100000
        if config.train_dataset_size is not None:
            iterations = math.ceil(
                config.train_dataset_size / max(config.batch_size, config.nominal_batch_size)
            ) * config.epochs
        class_count = int(getattr(model, "nc", 10))
        fitted_lr = round(0.002 * 5 / (4 + class_count), 6)
        if iterations > 10000:
            name, learning_rate, momentum = "sgd", 0.01, 0.9
        else:
            name, learning_rate, momentum = "adamw", fitted_lr, 0.9
            config.warmup_bias_lr = 0.0

    decay = (
        config.weight_decay
        * config.batch_size
        * config.gradient_accumulation_steps
        / config.nominal_batch_size
    )
    parameter_groups_list = []
    for group_name, group_decay in (("bias", 0.0), ("decay", decay), ("norm", 0.0)):
        if parameter_groups[group_name]:
            parameter_groups_list.append({
                "params": parameter_groups[group_name],
                "weight_decay": group_decay,
                "group_name": group_name,
            })

    config.resolved_optimizer = name.upper()
    config.resolved_learning_rate = float(learning_rate)
    config.resolved_momentum = float(momentum)
    config.resolved_weight_decay = float(decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameter_groups_list, lr=learning_rate, momentum=momentum, nesterov=True
        )
    if name == "adam":
        return torch.optim.Adam(parameter_groups_list, lr=learning_rate, betas=(momentum, 0.999))
    if name == "adamw":
        return torch.optim.AdamW(parameter_groups_list, lr=learning_rate, betas=(momentum, 0.999))
    if name == "adamax":
        return torch.optim.Adamax(parameter_groups_list, lr=learning_rate, betas=(momentum, 0.999))
    if name == "nadam":
        return torch.optim.NAdam(parameter_groups_list, lr=learning_rate, betas=(momentum, 0.999))
    if name == "radam":
        return torch.optim.RAdam(parameter_groups_list, lr=learning_rate, betas=(momentum, 0.999))
    if name == "rmsprop":
        return torch.optim.RMSprop(parameter_groups_list, lr=learning_rate, momentum=momentum)
    raise ValueError("optimizer must be one of: auto, SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp")


def build_scheduler(optimizer, config: TrainingConfig):
    """Create a linear or cosine epoch schedule from the project training settings."""
    torch = _require_torch()
    if config.lr_milestones is not None:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.lr_milestones),
            gamma=float(config.lr_gamma),
        )

    def scale(epoch_index: int) -> float:
        progress = min(max(epoch_index, 0) / config.epochs, 1.0)
        if config.cosine_schedule:
            return config.final_lr_factor + (1.0 - config.final_lr_factor) * (
                1.0 + math.cos(math.pi * progress)
            ) * 0.5
        return 1.0 - (1.0 - config.final_lr_factor) * progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def freeze_model_layers(model, freeze) -> Tuple[list, int]:
    """Freeze the requested graph layers and the fixed DFL projection layer."""
    graph = getattr(model, "model", None)
    layer_count = len(graph) if graph is not None else 0
    if freeze is None:
        layer_indices = []
    elif isinstance(freeze, int) and not isinstance(freeze, bool):
        if freeze > layer_count:
            raise ValueError(f"freeze={freeze} exceeds the model's {layer_count} layers")
        layer_indices = list(range(freeze))
    elif isinstance(freeze, (list, tuple)):
        layer_indices = list(freeze)
        invalid = [index for index in layer_indices if index >= layer_count]
        if invalid:
            raise ValueError(f"freeze layer indices {invalid} exceed the model's {layer_count} layers")
    else:
        raise ValueError("freeze must be null, an integer layer count, or a list of layer indices")
    if layer_indices and graph is None:
        raise TypeError("This model does not expose indexed layers required by freeze")

    freeze_prefixes = [f"model.{index}." for index in layer_indices]
    frozen_count = 0
    for name, parameter in model.named_parameters():
        should_freeze = any(name.startswith(prefix) for prefix in freeze_prefixes) or ".dfl." in name
        if should_freeze and parameter.requires_grad:
            parameter.requires_grad = False
            frozen_count += parameter.numel()
    return freeze_prefixes + [".dfl"], frozen_count


def _move_to_device(value, device, torch):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, torch) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device, torch) for item in value]
    return value


def _split_batch(batch, device, torch) -> Tuple[Any, Any]:
    """Accept the shared mapping form or the conventional ``(images, targets)`` pair."""
    if isinstance(batch, Mapping):
        if "images" in batch:
            images = batch["images"]
        elif "image" in batch:
            images = batch["image"]
        else:
            raise ValueError("A mapping batch must contain an 'images' or 'image' field")
        if "targets" in batch:
            targets = batch["targets"]
        else:
            targets = {key: value for key, value in batch.items() if key not in ("images", "image")}
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, targets = batch
    else:
        raise TypeError("A training batch must be a mapping or an (images, targets) pair")
    return _move_to_device(images, device, torch), _move_to_device(targets, device, torch)


def _loss_components(result, torch):
    if torch.is_tensor(result):
        if result.numel() != 1:
            raise TypeError("The criterion loss tensor must contain exactly one value")
        return result, {"loss": result}
    if isinstance(result, Mapping):
        if (
            "loss" not in result
            or not torch.is_tensor(result["loss"])
            or result["loss"].numel() != 1
        ):
            raise TypeError("A loss mapping must contain a scalar tensor under the 'loss' key")
        metric_values = result.get("metrics")
        metric_values = metric_values if isinstance(metric_values, Mapping) else result
        values = {}
        for key, value in metric_values.items():
            if torch.is_tensor(value) and value.numel() == 1:
                values[str(key)] = value
        return result["loss"], values
    raise TypeError("The criterion must return a loss tensor or a mapping containing 'loss'")


class Trainer:
    """Run training with pluggable data, criteria, validation and progress callbacks.

    The loop owns optimizer steps, mixed precision, resume state, checkpoints,
    early stopping and event emission. Model-specific output decoding and loss
    calculation stay in the supplied ``criterion`` and ``validator``.
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
        optimizer=None,
        scheduler=None,
        events: Optional[EventEmitter] = None,
    ) -> None:
        torch = _require_torch()
        self.torch = torch
        self.config = config
        self.distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if self.distributed else 0
        self.world_size = torch.distributed.get_world_size() if self.distributed else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        seed_everything(config.seed + self.rank, config.deterministic)
        self.device = _resolve_device(
            torch, config.device, local_rank=self.local_rank, distributed=self.distributed
        )
        self.raw_model = model.to(self.device)
        self.freeze_layer_names, self.frozen_parameter_count = freeze_model_layers(
            self.raw_model, config.freeze
        )
        self.optimizer = optimizer or build_optimizer(self.raw_model, config)
        self.scheduler = scheduler or build_scheduler(self.optimizer, config)
        self.ema = ModelEMA(self.raw_model, enabled=config.ema)
        self.model = (
            torch.nn.parallel.DistributedDataParallel(
                self.raw_model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
            )
            if self.distributed else self.raw_model
        )
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.scaler = _make_grad_scaler(torch, enabled=self.amp_enabled)
        self.events = (events or EventEmitter()) if self.rank == 0 else EventEmitter()
        self._stop_requested = threading.Event()
        self.start_epoch = 0
        self.completed_epochs = 0
        self.global_step = 0
        self.best_metric = None
        self._best_epochs_without_improvement = 0
        self._warmup_steps = -1
        self._last_opt_step = -1
        self._iteration_count = 0
        self._history = []
        self._time_deadline = None
        self._time_limit_reached = False

        if config.resume_from is not None:
            self.resume(config.resume_from)

    def request_stop(self) -> None:
        """Ask the loop to stop after its current optimizer step and save state."""
        self._stop_requested.set()

    def _model_train(self) -> None:
        """Set training mode while keeping BatchNorm state frozen in frozen layers."""
        self.model.train()
        if not self.config.freeze:
            return
        batch_norm = self.torch.nn.BatchNorm2d
        for name, module in self.raw_model.named_modules():
            if any(name.startswith(prefix) for prefix in self.freeze_layer_names) and isinstance(
                module, batch_norm
            ):
                module.eval()

    def _multi_scale(self, images):
        """Randomly rescale training batches using stride-aligned YOLO sizing."""
        if not self.config.multi_scale:
            return images
        head = getattr(self.raw_model, "model", None)
        head = head[-1] if head is not None and len(head) else None
        strides = getattr(head, "stride", None)
        stride = max(int(strides.max().item()), 1) if strides is not None else 32
        size = random.randrange(
            int(self.config.image_size * 0.5), int(self.config.image_size * 1.5 + stride)
        ) // stride * stride
        factor = size / max(images.shape[-2:])
        if factor == 1:
            return images
        new_shape = [
            math.ceil(dimension * factor / stride) * stride
            for dimension in images.shape[-2:]
        ]
        return self.torch.nn.functional.interpolate(
            images, size=new_shape, mode="bilinear", align_corners=False
        )

    def resume(self, checkpoint_path) -> None:
        metadata = load_checkpoint(
            checkpoint_path,
            self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
            ema_model=self.ema.model if self.config.ema else None,
        )
        self.start_epoch = metadata["epoch"]
        self.completed_epochs = metadata["epoch"]
        self.global_step = metadata["global_step"]
        self.best_metric = metadata["best_metric"]
        self._best_epochs_without_improvement = metadata["epochs_without_improvement"]

    def fit(
        self,
        train_loader: Iterable,
        criterion: Callable,
        validation_loader: Optional[Iterable] = None,
        validator: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Train until the configured epoch count, early stop or stop request."""
        if validation_loader is not None and validator is None:
            raise ValueError("validator must be supplied when a validation_loader is provided")
        if validation_loader is None and validator is not None:
            raise ValueError("validation_loader must be supplied when a validator is provided")
        if hasattr(criterion, "to"):
            criterion.to(self.device)
        self._time_deadline = (
            time.monotonic() + self.config.time_hours * 3600.0
            if self.config.time_hours is not None else None
        )

        self.events.emit(TrainingEvent(
            event="started",
            epoch=self.start_epoch,
            epochs=self.config.epochs,
            message=f"device={self.device}",
        ))
        if self.frozen_parameter_count:
            self.events.emit(TrainingEvent(
                event="freeze",
                metrics={"frozen_parameters": float(self.frozen_parameter_count)},
                message=f"Froze {self.frozen_parameter_count} model parameters",
            ))
        final_metrics: Dict[str, float] = {}
        stopped = False
        succeeded = False
        best_path = self.config.output_dir / "weights" / "best.pt"
        last_path = self.config.output_dir / "weights" / "last.pt"
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        if self.config.warmup_iters is not None:
            self._warmup_steps = max(int(self.config.warmup_iters), 0)
        elif total_batches and total_batches > 0 and self.config.warmup_epochs > 0:
            warmup_epochs = min(
                float(self.config.warmup_epochs), max(self.config.epochs - 1, 0)
            )
            self._warmup_steps = int(round(warmup_epochs * total_batches))
        else:
            self._warmup_steps = 0
        self.optimizer.zero_grad(set_to_none=True)
        self._last_opt_step = -1
        self._iteration_count = (
            self.start_epoch * total_batches if total_batches and total_batches > 0 else 0
        )

        try:
            for epoch_index in range(self.start_epoch, self.config.epochs):
                if self._time_deadline is not None and time.monotonic() >= self._time_deadline:
                    self._time_limit_reached = True
                    stopped = True
                    break
                epoch_number = epoch_index + 1
                self._set_epoch(train_loader, epoch_index)
                loader_generator = getattr(train_loader, "generator", None)
                if loader_generator is not None:
                    loader_generator.manual_seed(
                        int(self.config.seed) + self.rank + int(epoch_index)
                    )
                train_metrics, interrupted = self._train_epoch(train_loader, criterion, epoch_number)
                final_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
                if interrupted:
                    stopped = True
                    break

                if self.scheduler is not None and not isinstance(
                    self.scheduler, self.torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

                should_validate = (
                    validation_loader is not None
                    and epoch_number % self.config.validation_interval == 0
                )
                if should_validate:
                    validation_metrics = self._validate(
                        validation_loader, validator, epoch_number
                    )
                    final_metrics.update({f"val_{key}": value for key, value in validation_metrics.items()})

                    monitored = final_metrics.get(self.config.best_metric)
                    if monitored is None:
                        monitored = validation_metrics.get(self.config.best_metric)
                    if monitored is None:
                        raise KeyError(
                            f"best_metric '{self.config.best_metric}' was not returned by validation"
                        )
                    improved = self._is_improved(float(monitored))
                    if improved:
                        self.best_metric = float(monitored)
                        self._best_epochs_without_improvement = 0
                        self._save_checkpoint(
                            best_path, self.optimizer, self.scheduler, self.scaler,
                            epoch_number, self.global_step, self.best_metric,
                            self._best_epochs_without_improvement, self.config,
                            ema_model=self.ema.model if self.config.ema else None,
                        )
                    else:
                        self._best_epochs_without_improvement += 1
                elif validation_loader is None:
                    monitored = train_metrics.get("loss")
                    if monitored is not None and self._is_improved(float(monitored), mode="min"):
                        self.best_metric = float(monitored)
                        self._best_epochs_without_improvement = 0
                        self._save_checkpoint(
                            best_path, self.optimizer, self.scheduler, self.scaler,
                            epoch_number, self.global_step, self.best_metric,
                            self._best_epochs_without_improvement, self.config,
                            ema_model=self.ema.model if self.config.ema else None,
                        )

                if self.scheduler is not None:
                    if isinstance(self.scheduler, self.torch.optim.lr_scheduler.ReduceLROnPlateau):
                        metric = None
                        if should_validate:
                            metric = validation_metrics.get(self.config.best_metric)
                        elif validation_loader is None:
                            metric = train_metrics.get("loss")
                        if metric is not None:
                            self.scheduler.step(metric)

                self._save_checkpoint(
                    last_path, self.optimizer, self.scheduler, self.scaler,
                    epoch_number, self.global_step, self.best_metric,
                    self._best_epochs_without_improvement, self.config,
                    ema_model=self.ema.model if self.config.ema else None,
                )
                if self.config.save_period and epoch_number % self.config.save_period == 0:
                    self._save_checkpoint(
                        self.config.output_dir / "weights" / f"epoch-{epoch_number}.pt",
                        self.optimizer, self.scheduler, self.scaler,
                        epoch_number, self.global_step, self.best_metric,
                        self._best_epochs_without_improvement, self.config,
                        ema_model=self.ema.model if self.config.ema else None,
                    )
                self.completed_epochs = epoch_number

                self.events.emit(TrainingEvent(
                    event="epoch_end",
                    epoch=epoch_number,
                    epochs=self.config.epochs,
                    metrics=final_metrics,
                ))
                self._history.append({"epoch": epoch_number, **final_metrics})

                if self._stop_requested.is_set():
                    stopped = True
                    break
                if (
                    should_validate
                    and self.config.early_stopping_patience > 0
                    and self._best_epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    stopped = True
                    break

            succeeded = True
            return {
                "epoch": self.completed_epochs,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "metrics": final_metrics,
                "stopped": stopped and not self._time_limit_reached,
                "time_limit_reached": self._time_limit_reached,
            }
        except Exception as exc:
            self.events.emit(TrainingEvent(event="failed", message=str(exc)))
            raise
        finally:
            if self._time_limit_reached:
                self._save_checkpoint(
                    last_path, self.optimizer, self.scheduler, self.scaler,
                    self.completed_epochs, self.global_step, self.best_metric,
                    self._best_epochs_without_improvement, self.config,
                    ema_model=self.ema.model if self.config.ema else None,
                )
                self.events.emit(TrainingEvent(
                    event="finished",
                    epoch=self.completed_epochs,
                    epochs=self.config.epochs,
                    message=(
                        "Configured training time limit reached; checkpoint saved."
                        if self.config.save else
                        "Configured training time limit reached; checkpoint saving is disabled."
                    ),
                ))
            elif stopped or self._stop_requested.is_set():
                # Keep the epoch counter at the last completed epoch. Resume will
                # replay an interrupted epoch with the saved model and optimizer state.
                self._save_checkpoint(
                    last_path, self.optimizer, self.scheduler, self.scaler,
                    self.completed_epochs, self.global_step, self.best_metric,
                    self._best_epochs_without_improvement, self.config,
                    ema_model=self.ema.model if self.config.ema else None,
                )
                self.events.emit(TrainingEvent(
                    event="stopped",
                    epoch=self.completed_epochs,
                    epochs=self.config.epochs,
                    message="Training stopped at the requested safe point.",
                ))
            elif succeeded:
                self.events.emit(TrainingEvent(
                    event="finished",
                    epoch=self.config.epochs,
                    epochs=self.config.epochs,
                ))
            if self.config.save and self.config.plots and self.rank == 0:
                self._write_training_plots()

    def _train_epoch(self, loader, criterion, epoch_number: int) -> Tuple[Dict[str, float], bool]:
        torch = self.torch
        self._model_train()
        if hasattr(criterion, "train"):
            criterion.train()
        totals: Dict[str, float] = {}
        batches = 0
        profiling = bool(self.config.profile)
        profile_totals = {
            "data_wait_ms": 0.0,
            "forward_loss_ms": 0.0,
            "backward_step_ms": 0.0,
        }
        epoch_started = time.perf_counter()
        wait_started = epoch_started

        def synchronize_for_profile():
            if profiling and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        if total_batches is not None and total_batches < 1:
            total_batches = None

        for batch_index, batch in enumerate(loader):
            if self._stop_requested.is_set() and not self.distributed:
                break
            if profiling:
                synchronize_for_profile()
                forward_started = time.perf_counter()
                profile_totals["data_wait_ms"] += (forward_started - wait_started) * 1000.0
            if total_batches is not None:
                iteration = batch_index + (epoch_number - 1) * total_batches
            else:
                iteration = self._iteration_count
                self._iteration_count += 1
            accumulation = self.config.gradient_accumulation_steps
            if self._warmup_steps > 0 and iteration < self._warmup_steps:
                progress = min(max(iteration / self._warmup_steps, 0.0), 1.0)
                target_accumulation = (
                    self.config.nominal_batch_size / self.config.batch_size
                )
                accumulation = max(1, int(round(1.0 + (target_accumulation - 1.0) * progress)))
                for group_index, group in enumerate(self.optimizer.param_groups):
                    if self.config.warmup_start_factor is not None:
                        warmup_start = (
                            group.get("initial_lr", group["lr"])
                            * self.config.warmup_start_factor
                        )
                    else:
                        warmup_start = (
                            self.config.warmup_bias_lr if group.get("group_name") == "bias" else 0.0
                        )
                    if self.scheduler is not None and hasattr(self.scheduler, "lr_lambdas"):
                        schedule_factor = self.scheduler.lr_lambdas[group_index](epoch_number - 1)
                    elif self.scheduler is not None and hasattr(self.scheduler, "milestones"):
                        milestone_counts = getattr(self.scheduler, "milestones")
                        gamma = float(getattr(self.scheduler, "gamma", 1.0))
                        decay_steps = sum(
                            count
                            for milestone, count in milestone_counts.items()
                            if milestone <= epoch_number
                        )
                        schedule_factor = gamma ** decay_steps
                    else:
                        schedule_factor = 1.0
                    target_lr = group.get("initial_lr", group["lr"]) * schedule_factor
                    group["lr"] = warmup_start + (target_lr - warmup_start) * progress
                    if "momentum" in group:
                        group["momentum"] = (
                            self.config.warmup_momentum
                            + (self.config.momentum - self.config.warmup_momentum) * progress
                        )
            images, targets = _split_batch(batch, self.device, torch)
            images = self._multi_scale(images)
            autocast = (
                torch.autocast(device_type="cuda", enabled=True)
                if self.amp_enabled else nullcontext()
            )
            with autocast:
                predictions = self.model(images)
                loss, components = _loss_components(criterion(predictions, targets), torch)
                if getattr(criterion, "loss_is_batch_sum", False):
                    # DDP averages gradients across ranks; restore the global batch-sum
                    # scale so multi-GPU training matches the single-process objective.
                    scaled_loss = loss * (self.world_size if self.distributed else 1)
                else:
                    scaled_loss = loss / accumulation
            if profiling:
                synchronize_for_profile()
                profile_totals["forward_loss_ms"] += (time.perf_counter() - forward_started) * 1000.0

            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch_number}, batch {batch_index + 1}"
                )
            if profiling:
                backward_started = time.perf_counter()
            self.scaler.scale(scaled_loss).backward()
            should_step = iteration - self._last_opt_step >= accumulation
            if should_step:
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.raw_model)
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._last_opt_step = iteration
            if profiling:
                synchronize_for_profile()
                profile_totals["backward_step_ms"] += (time.perf_counter() - backward_started) * 1000.0

            batches += 1
            for name, component in components.items():
                totals[name] = totals.get(name, 0.0) + float(component.detach().float().item())
            if self.config.verbose and batch_index % self.config.log_interval == 0:
                self.events.emit(TrainingEvent(
                    event="batch",
                    epoch=epoch_number,
                    epochs=self.config.epochs,
                    step=batch_index + 1,
                    steps=total_batches,
                    metrics={name: float(component.detach().float().item())
                             for name, component in components.items()},
                ))
            if profiling:
                wait_started = time.perf_counter()
            if self._time_deadline is not None and time.monotonic() >= self._time_deadline:
                self._time_limit_reached = True
                self._stop_requested.set()
            if self.distributed:
                stop_value = torch.tensor(
                    [2 if self._time_limit_reached else 1 if self._stop_requested.is_set() else 0],
                    dtype=torch.int32,
                    device=self.device,
                )
                torch.distributed.all_reduce(stop_value, op=torch.distributed.ReduceOp.MAX)
                if int(stop_value.item()):
                    self._stop_requested.set()
                    self._time_limit_reached = int(stop_value.item()) == 2
                    break

        if batches == 0 and not self._stop_requested.is_set():
            raise ValueError("The training data loader produced no batches")
        if profiling:
            synchronize_for_profile()
            totals.update({
                f"profile/{name}": value
                for name, value in profile_totals.items()
            })
            totals["profile/epoch_ms"] = (time.perf_counter() - epoch_started) * 1000.0
        if self.distributed:
            names = sorted(totals)
            packed = torch.tensor(
                [*(totals[name] for name in names), float(batches)],
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            totals = {name: float(packed[index].item()) for index, name in enumerate(names)}
            batches = int(packed[-1].item())
        average_metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        if profiling:
            average_metrics["profile/epoch_ms"] = totals["profile/epoch_ms"] / max(self.world_size, 1)
        return average_metrics, self._stop_requested.is_set()

    def _validate(self, loader, validator: Callable, epoch_number: int) -> Dict[str, float]:
        self.ema.model.eval()
        if hasattr(validator, "eval"):
            validator.eval()
        with self.torch.inference_mode():
            result = validator(self.ema.model, loader, self.device)
        if not isinstance(result, Mapping):
            raise TypeError("validator must return a mapping of metric names to scalar values")
        metrics = {str(key): float(value) for key, value in result.items()}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Validation returned a non-finite metric at epoch {epoch_number}")
        self.events.emit(TrainingEvent(
            event="validation",
            epoch=epoch_number,
            epochs=self.config.epochs,
            metrics=metrics,
        ))
        return metrics

    def _is_improved(self, value: float, mode: Optional[str] = None) -> bool:
        if self.best_metric is None:
            return True
        if (mode or self.config.best_metric_mode) == "max":
            return value > self.best_metric
        return value < self.best_metric

    def _save_checkpoint(
        self,
        path,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step,
        best_metric,
        epochs_without_improvement,
        config,
        ema_model=None,
    ) -> None:
        if self.rank != 0 or not self.config.save:
            return
        save_checkpoint(
            path,
            self.raw_model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_metric,
            epochs_without_improvement,
            config,
            ema_model=ema_model,
        )

    @staticmethod
    def _set_epoch(loader, epoch_index: int) -> None:
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch_index)
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)

    def _write_training_plots(self) -> None:
        """Save local training and validation curves to ``results.png``."""
        if not self._history:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [item["epoch"] for item in self._history]
            if any("train_loss_simcc_x" in item for item in self._history):
                charts = (
                    ("Loss", ("train_loss",)),
                    ("SimCC X loss", ("train_loss_simcc_x",)),
                    ("SimCC Y loss", ("train_loss_simcc_y",)),
                    ("COCO AP", ("val_coco/AP", "val_coco/AP50", "val_coco/AP75")),
                    ("PCK / AUC", ("val_PCK", "val_AUC")),
                    ("EPE", ("val_EPE",)),
                )
            else:
                charts = (
                    ("Loss", ("train_loss", "val_loss")),
                    ("Box loss", ("train_box", "val_box")),
                    ("Class loss", ("train_class", "val_class")),
                    ("Pose loss", ("train_pose", "val_pose")),
                    ("COCO keypoint AP", ("val_coco/AP", "val_coco/AP50", "val_coco/AP75")),
                    ("COCO keypoint AR", ("val_coco/AR",)),
                    ("PCK / AUC", ("val_PCK", "val_AUC")),
                    ("EPE", ("val_EPE",)),
                )
            columns = 3 if len(charts) <= 6 else 4
            rows = (len(charts) + columns - 1) // columns
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(4.7 * columns, 4 * rows),
                constrained_layout=True,
                squeeze=False,
            )
            flat_axes = list(axes.flat)
            for axis, (title, keys) in zip(flat_axes, charts):
                plotted = False
                for key in keys:
                    values = [item.get(key) for item in self._history]
                    valid = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
                    if valid:
                        axis.plot(
                            [item[0] for item in valid],
                            [item[1] for item in valid],
                            marker="o",
                            label=key.replace("train_", "train ").replace("val_", "val "),
                        )
                        plotted = True
                axis.set_title(title)
                axis.set_xlabel("Epoch")
                axis.grid(True, alpha=0.25)
                if plotted:
                    axis.legend(fontsize="small")
            for axis in flat_axes[len(charts):]:
                axis.remove()
            destination = self.config.output_dir / "results.png"
            figure.savefig(destination, dpi=150)
            plt.close(figure)
            self.events.emit(TrainingEvent(event="plot", message=f"Training curves saved to {destination}"))
        except Exception as exc:
            self.events.emit(TrainingEvent(event="warning", message=f"Could not write training plots: {exc}"))
