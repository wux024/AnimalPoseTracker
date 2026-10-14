"""Model defaults and project-config adapters for the canonical training YAML."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from animalposetracker.cfg import MODEL_YAML_PATHS, TRAINING_CFG_PATH
from animalposetracker.data.topdown_defaults import (
    ANIMALVITPOSE_AUGMENTATION,
    ANIMALVITPOSE_PREPROCESSING,
)


with Path(TRAINING_CFG_PATH).open("r", encoding="utf-8") as _stream:
    TRAINING_PROFILES = yaml.safe_load(_stream) or {}
with Path(MODEL_YAML_PATHS["AnimalViTPose"]).open("r", encoding="utf-8") as _stream:
    ANIMALVITPOSE_MODEL_DEFAULTS = yaml.safe_load(_stream) or {}

_MODEL_PROFILES = TRAINING_PROFILES.get("models", {})
ANIMALRTPOSE_PROFILE = _MODEL_PROFILES.get("AnimalRTPose", {})
ANIMALVITPOSE_PROFILE = _MODEL_PROFILES.get("AnimalViTPose", {})
ANIMALVITPOSE_VARIANTS = dict(ANIMALVITPOSE_MODEL_DEFAULTS.get("model_variants", {}))
ANIMALVITPOSE_SIMCC = dict(ANIMALVITPOSE_MODEL_DEFAULTS.get("simcc", {}))
ANIMALVITPOSE_VALIDATION = dict(ANIMALVITPOSE_PROFILE.get("validation", {}))
KEYPOINT_METRICS = dict(TRAINING_PROFILES.get("metrics", {}))


def normalize_animalvitpose_scale(scale: str) -> str:
    """Resolve project model-scale aliases to the MMPose AnimalViTPose variants."""
    aliases = {
        "n": "small",
        "s": "small",
        "small": "small",
        "b": "base",
        "base": "base",
        "l": "large",
        "large": "large",
        "h": "huge",
        "huge": "huge",
    }
    key = str(scale or "small").strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported AnimalViTPose scale {scale!r}; choose small, base, large or huge"
        )
    return aliases[key]


def configure_animalvitpose_model(model_spec: Dict[str, Any], scale: str, image_size: int) -> str:
    """Apply an MMPose ViT variant and square input size to the local YAML graph."""
    variant_name = normalize_animalvitpose_scale(scale)
    variants = model_spec.get("model_variants") or ANIMALVITPOSE_VARIANTS
    variant = variants[variant_name]
    simcc = model_spec.get("simcc") or ANIMALVITPOSE_SIMCC
    input_size = int(image_size)
    if input_size < 16 or input_size % 16:
        raise ValueError("AnimalViTPose image_size must be a positive multiple of patch_size=16")

    backbone_entries = model_spec.get("backbone") or []
    if not isinstance(backbone_entries, list) or not backbone_entries:
        raise ValueError("AnimalViTPose model configuration must contain a ViT backbone")
    backbone_entry = backbone_entries[0]
    if not isinstance(backbone_entry, list) or len(backbone_entry) < 4 or str(backbone_entry[2]) != "ViT":
        raise ValueError("AnimalViTPose backbone must be the registered ViT module")
    backbone_args = backbone_entry[3]
    if not isinstance(backbone_args, list) or len(backbone_args) < 6:
        raise ValueError("AnimalViTPose ViT configuration has incomplete architecture arguments")
    backbone_args[:4] = [
        variant["embed_dims"],
        variant["num_layers"],
        variant["num_heads"],
        variant["feedforward_channels"],
    ]
    patch_size = int(backbone_args[4])
    if input_size % patch_size:
        raise ValueError(f"AnimalViTPose image_size={input_size} must be divisible by patch_size={patch_size}")
    backbone_args[5] = input_size
    if len(backbone_args) < 7:
        backbone_args.append(None)
    if len(backbone_args) < 8:
        backbone_args.append({})
    options = dict(backbone_args[7] or {})
    options["drop_path_rate"] = variant["drop_path_rate"]
    backbone_args[7] = options

    head_entries = model_spec.get("head") or []
    if not isinstance(head_entries, list) or not head_entries:
        raise ValueError("AnimalViTPose model configuration must contain a SimCCHead")
    head_entry = head_entries[-1]
    if not isinstance(head_entry, list) or len(head_entry) < 4 or str(head_entry[2]) != "SimCCHead":
        raise ValueError("AnimalViTPose final model layer must be SimCCHead")
    head_args = head_entry[3]
    if not isinstance(head_args, list) or len(head_args) < 3:
        raise ValueError("AnimalViTPose SimCCHead configuration has incomplete arguments")
    head_args[1] = [input_size, input_size]
    head_args[2] = [input_size // patch_size, input_size // patch_size]
    if len(head_args) > 3:
        head_args[3] = float(simcc.get("split_ratio", head_args[3]))
    return variant_name


def apply_animalvitpose_recipe(
    values: Dict[str, Any],
    model_scale: str = "small",
    model_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], list]:
    """Fill the two ViTPose recipe values that are not plain defaults.

    The recipe's hyper-parameters now live in training.yaml's ViTPose profile
    (resolved before this runs), so only the two dynamic pieces remain: the
    per-scale layer-decay default and the MAE pretrained URL. Both are filled
    only when the value is still absent -- a user-given value is never
    rewritten, which retires the old "value == generic default" heuristic
    (the one that once rewrote an explicit ``--epochs 1000`` into 500).
    """
    resolved = dict(values or {})
    scale_name = normalize_animalvitpose_scale(model_scale)
    applied = []

    # Layer decay is an optimizer choice, so its per-size defaults stay in the
    # training profile instead of being mixed into the model architecture yaml.
    if resolved.get("layer_decay_rate") is None:
        profile_training = dict(ANIMALVITPOSE_PROFILE.get("training", {}) or {})
        by_scale = profile_training.get("layer_decay_rate_by_scale") or {}
        if scale_name in by_scale:
            resolved["layer_decay_rate"] = by_scale[scale_name]
            applied.append("layer_decay_rate")

    # The MMPose configs initialize the ViT backbone from the matching MAE
    # checkpoint; ``pretrained: false`` explicitly disables it.
    if not resolved.get("pretrained_weights"):
        pretrained_setting = resolved.get("pretrained")
        if isinstance(pretrained_setting, str) and pretrained_setting.strip():
            resolved["pretrained_weights"] = pretrained_setting.strip()
            applied.append("pretrained_weights")
        elif pretrained_setting is not False:
            config = model_config or ANIMALVITPOSE_MODEL_DEFAULTS
            variants = config.get("model_variants") or ANIMALVITPOSE_VARIANTS
            variant = variants[scale_name]
            resolved["pretrained_weights"] = variant["pretrained_url"]
            applied.append("pretrained_weights")
    return resolved, applied


def train_scope_prefixes(model_type: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    """The train section's flattening order, last one wins.

    Shared settings first, then the active model's profile; ``metrics`` keys
    ride along at the end so the evaluator reads them from the same mapping.
    """
    model = "AnimalViTPose" if str(model_type or "").casefold() == "animalvitpose" else "AnimalRTPose"
    return (
        ("shared.runtime", "train.shared.runtime."),
        ("shared.training", "train.shared.training."),
        (f"models.{model}.training", f"train.models.{model}.training."),
        (f"models.{model}.loss", f"train.models.{model}.loss."),
        (f"models.{model}.augmentation", f"train.models.{model}.augmentation."),
        (f"models.{model}.validation", f"train.models.{model}.validation."),
        ("metrics", "train.metrics."),
    )


def flatten_train_values(resolved, model_type: Optional[str]):
    """Flatten a resolved train scope for the trainer.

    Returns ``(values, origins)``: the canonical spellings the trainer's
    config object consumes, and -- per key -- the dotted path the winning
    value came from (the command line is re-anchored through these origins,
    so an override lands on the section that actually decides). The four
    shared keys (device / project / name / exist_ok) ride along from the
    ``common`` section.
    """
    values: Dict[str, Any] = {}
    origins: Dict[str, str] = {}
    for _group, prefix in train_scope_prefixes(model_type):
        for path, entry in resolved.values.items():
            if not path.startswith(prefix):
                continue
            key = path[len(prefix):]
            if "." in key:
                continue  # only leaves flatten; nested tables stay tables
            values[key] = entry.value
            origins[key] = path
    for key in ("device", "project", "name", "exist_ok"):
        path = f"common.{key}"
        entry = resolved.values.get(path)
        if entry is not None:
            values[key] = entry.value
            origins[key] = path
    return values, origins


def train_key_targets(model_type: Optional[str]) -> Dict[str, str]:
    """Canonical flat key -> the dotted path that decides it (last one wins).

    Used to anchor command-line overrides and to place creation-time values
    into the project's runtime tree.
    """
    from animalposetracker.config.schema import fields

    model = "AnimalViTPose" if str(model_type or "").casefold() == "animalvitpose" else "AnimalRTPose"
    targets: Dict[str, str] = {}
    for field in fields(scope="train"):
        if field.group == "shared.runtime":
            targets[field.key] = f"train.shared.runtime.{field.key}"
        elif field.group == "shared.training":
            targets[field.key] = f"train.shared.training.{field.key}"
        elif field.group == "metrics":
            targets[field.key] = f"train.metrics.{field.key}"
        elif field.group.startswith("models."):
            section = field.group.split(".", 1)[1]
            targets[field.key] = f"train.models.{model}.{section}.{field.key}"
    return targets


__all__ = [
    "ANIMALVITPOSE_VARIANTS",
    "ANIMALVITPOSE_MODEL_DEFAULTS",
    "ANIMALVITPOSE_SIMCC",
    "ANIMALVITPOSE_PREPROCESSING",
    "ANIMALVITPOSE_AUGMENTATION",
    "ANIMALVITPOSE_VALIDATION",
    "KEYPOINT_METRICS",
    "ANIMALRTPOSE_PROFILE",
    "apply_animalvitpose_recipe",
    "configure_animalvitpose_model",
    "flatten_train_values",
    "train_key_targets",
    "train_scope_prefixes",
    "normalize_animalvitpose_scale",
]
