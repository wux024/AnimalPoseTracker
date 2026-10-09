"""Model defaults and project-config adapters for the canonical training YAML."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml

from animalposetracker.cfg import MODEL_YAML_PATHS, TRAINING_CFG_PATH


with Path(TRAINING_CFG_PATH).open("r", encoding="utf-8") as _stream:
    TRAINING_PROFILES = yaml.safe_load(_stream) or {}
with Path(MODEL_YAML_PATHS["AnimalViTPose"]).open("r", encoding="utf-8") as _stream:
    ANIMALVITPOSE_MODEL_DEFAULTS = yaml.safe_load(_stream) or {}

_SHARED_GROUPS = ("runtime", "training", "loss", "augmentation", "validation")
_PROFILE_SECTIONS = ("training", "loss", "augmentation", "validation")


def _shared_profile_values(training_tree: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten grouped common settings, while reading older flat shared mappings."""
    shared = dict(training_tree.get("shared", {}) or {})
    if not any(isinstance(shared.get(group), Mapping) for group in _SHARED_GROUPS):
        return shared

    values = {key: value for key, value in shared.items() if key not in _SHARED_GROUPS}
    for group in _SHARED_GROUPS:
        values.update(shared.get(group, {}) or {})
    return values


def _group_shared_settings(shared: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize flat shared settings into semantic YAML groups."""
    values = deepcopy(dict(shared or {}))
    if not any(isinstance(values.get(group), Mapping) for group in _SHARED_GROUPS):
        return {"runtime": values}

    grouped = {
        group: deepcopy(dict(values.get(group, {}) or {}))
        for group in _SHARED_GROUPS
        if isinstance(values.get(group), Mapping)
    }
    extras = {key: value for key, value in values.items() if key not in _SHARED_GROUPS}
    if extras:
        grouped.setdefault("runtime", {}).update(extras)
    return grouped


def _consolidate_equal_model_defaults(training_tree: Dict[str, Any]) -> None:
    """Move identical per-model values to shared defaults; retain differing overrides."""
    profiles = training_tree.get("models", {}) or {}
    profile_values = [profile or {} for profile in profiles.values()]
    if len(profile_values) < 2:
        return

    shared = training_tree.setdefault("shared", {})
    for section in _PROFILE_SECTIONS:
        section_values = [profile.get(section, {}) or {} for profile in profile_values]
        common_keys = set(section_values[0])
        for values in section_values[1:]:
            common_keys.intersection_update(values)
        for key in common_keys:
            value = section_values[0][key]
            if all(values[key] == value for values in section_values[1:]):
                shared.setdefault(section, {})[key] = deepcopy(value)
                for values in section_values:
                    values.pop(key, None)

_PROJECT_KEY_ALIASES = {
    "batch_size": "batch",
    "image_size": "imgsz",
    "num_workers": "workers",
    "learning_rate": "lr0",
    "final_lr_factor": "lrf",
    "cosine_schedule": "cos_lr",
    "nominal_batch_size": "nbs",
    "early_stopping_patience": "patience",
    "validation_enabled": "val",
    "validation_confidence": "conf",
    "validation_iou": "iou",
    "max_detections": "max_det",
    "time_hours": "time",
    "box_loss_weight": "box",
    "class_loss_weight": "cls",
    "distribution_loss_weight": "dfl",
    "keypoint_loss_weight": "pose",
    "visibility_loss_weight": "kobj",
}

_MODEL_PROFILES = TRAINING_PROFILES.get("models", {})
ANIMALRTPOSE_PROFILE = _MODEL_PROFILES.get("AnimalRTPose", {})
ANIMALVITPOSE_PROFILE = _MODEL_PROFILES.get("AnimalViTPose", {})
ANIMALVITPOSE_RECIPE = {
    **_shared_profile_values(TRAINING_PROFILES),
    **dict(ANIMALVITPOSE_PROFILE.get("training", {})),
    **dict(ANIMALVITPOSE_PROFILE.get("loss", {})),
}
ANIMALVITPOSE_VARIANTS = dict(ANIMALVITPOSE_MODEL_DEFAULTS.get("model_variants", {}))
ANIMALVITPOSE_SIMCC = dict(ANIMALVITPOSE_MODEL_DEFAULTS.get("simcc", {}))
ANIMALVITPOSE_PREPROCESSING = dict(ANIMALVITPOSE_MODEL_DEFAULTS.get("preprocessing", {}))
ANIMALVITPOSE_AUGMENTATION = dict(ANIMALVITPOSE_PROFILE.get("augmentation", {}))
ANIMALVITPOSE_VALIDATION = dict(ANIMALVITPOSE_PROFILE.get("validation", {}))
KEYPOINT_METRICS = dict(TRAINING_PROFILES.get("metrics", {}))


def training_profile_defaults(model_type: str) -> Dict[str, Any]:
    """Flatten common and selected model defaults for project configs."""
    model_name = str(model_type).casefold()
    key = "AnimalViTPose" if model_name == "animalvitpose" else "AnimalRTPose"
    profile = _MODEL_PROFILES.get(key, {})
    values = _shared_profile_values(TRAINING_PROFILES)
    sections = ["training", "loss", "augmentation"]
    if key == "AnimalRTPose":
        sections.append("validation")
    for section in sections:
        values.update(profile.get(section, {}) or {})
    return values


def project_training_defaults(model_type: str) -> Dict[str, Any]:
    """Return canonical profile values in the flat key spelling used by project files."""
    values = training_profile_defaults(model_type)
    return {_PROJECT_KEY_ALIASES.get(key, key): value for key, value in values.items()}


def _profile_section_values(training_tree: Mapping[str, Any], model_type: str):
    profiles = training_tree.get("models", {}) or {}
    profile_name = "AnimalViTPose" if str(model_type).casefold() == "animalvitpose" else "AnimalRTPose"
    profile = profiles.get(profile_name, {}) or {}
    values = _shared_profile_values(training_tree)
    for section in _PROFILE_SECTIONS:
        values.update(profile.get(section, {}) or {})
    return values


def flatten_project_training_config(
    values: Mapping[str, Any],
    model_type: Optional[str] = None,
    include_inactive_model_fields: bool = False,
) -> Dict[str, Any]:
    """Flatten a hierarchical project's ``other.yaml`` for the trainer or GUI view.

    Flat project files remain readable as-is. New files keep shared defaults and both
    model profiles nested under ``training``; the active profile is flattened only in
    memory before a run starts.
    """
    raw = dict(values or {})
    training_tree = raw.get("training")
    if not isinstance(training_tree, Mapping):
        return {key: value for key, value in raw.items() if key != "task"}

    selected_model = (
        model_type
        or training_tree.get("active_model")
        or raw.get("model_type")
        or "AnimalRTPose"
    )
    result = dict(raw.get("application", {}) or {})
    result.update({
        key: value
        for key, value in raw.items()
        if key not in {"training", "application", "task"}
    })
    result.update(training_tree.get("metrics", {}) or {})

    if include_inactive_model_fields:
        profile_values = _shared_profile_values(training_tree)
        profiles = training_tree.get("models", {}) or {}
        for profile_name, profile in profiles.items():
            for section in _PROFILE_SECTIONS:
                profile_values.update((profile or {}).get(section, {}) or {})
        active_values = _profile_section_values(training_tree, selected_model)
        profile_values.update(active_values)
    else:
        profile_values = _profile_section_values(training_tree, selected_model)

    result.update({
        _PROJECT_KEY_ALIASES.get(key, key): value
        for key, value in profile_values.items()
    })
    return result


def materialize_project_training_config(
    flat_values: Mapping[str, Any],
    training_tree: Optional[Mapping[str, Any]],
    model_type: str,
) -> Dict[str, Any]:
    """Apply the project's flat editable values to the selected nested profile."""
    tree = deepcopy(dict(training_tree or TRAINING_PROFILES))
    # Model scale is defined by project.yaml and model.yaml, not a training option.
    tree.pop("model_scale", None)
    # Model selection already lives in project.yaml; don't mirror it in other.yaml.
    tree.pop("active_model", None)
    tree["shared"] = _group_shared_settings(tree.get("shared", {}) or {})
    _consolidate_equal_model_defaults(tree)
    profile_name = "AnimalViTPose" if str(model_type).casefold() == "animalvitpose" else "AnimalRTPose"
    profile = tree.setdefault("models", {}).setdefault(profile_name, {})
    flat = dict(flat_values or {})

    active_profile_keys = {
        _PROJECT_KEY_ALIASES.get(key, key)
        for section in _PROFILE_SECTIONS
        for key in (profile.get(section, {}) or {})
    }
    for target in tree["shared"].values():
        if not isinstance(target, dict):
            continue
        for key in list(target):
            project_key = _PROJECT_KEY_ALIASES.get(key, key)
            if project_key in flat and project_key not in active_profile_keys:
                target[key] = flat[project_key]

    for section in _PROFILE_SECTIONS:
        target = profile.setdefault(section, {})
        for key in list(target):
            project_key = _PROJECT_KEY_ALIASES.get(key, key)
            if project_key in flat:
                target[key] = flat[project_key]

    metrics = tree.setdefault("metrics", {})
    for key in list(metrics):
        if key in flat:
            metrics[key] = flat[key]

    return tree


def nested_other_config_from_flat(
    flat_values: Mapping[str, Any],
    training_tree: Optional[Mapping[str, Any]],
    model_type: str,
) -> Dict[str, Any]:
    """Serialize one project as application settings plus nested model profiles."""
    flat = dict(flat_values or {})
    training_tree = materialize_project_training_config(
        flat,
        training_tree,
        model_type=model_type,
    )
    training_keys = set()
    training_keys.update(
        _PROJECT_KEY_ALIASES.get(key, key)
        for key in _shared_profile_values(training_tree)
    )
    training_keys.update((training_tree.get("metrics", {}) or {}).keys())
    for profile in (training_tree.get("models", {}) or {}).values():
        for section in ("training", "loss", "augmentation", "validation"):
            training_keys.update(
                _PROJECT_KEY_ALIASES.get(key, key)
                for key in ((profile or {}).get(section, {}) or {})
            )

    # ``task`` was inherited from the old YOLO defaults, but the project model
    # configuration already determines the pose task and the custom trainer
    # does not consume this field.
    root_keys = {"mode", "model", "data"}
    root = {key: value for key, value in flat.items() if key in root_keys}
    application = {
        key: value
        for key, value in flat.items()
        if key not in root_keys and key not in training_keys and key != "task"
    }
    return {**root, "application": application, "training": training_tree}


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


_ALIASES = {
    "batch_size": ("batch_size", "batch"),
    "image_size": ("image_size", "imgsz"),
    "learning_rate": ("learning_rate", "lr0"),
    "num_workers": ("num_workers", "workers"),
    "validation_interval": ("validation_interval", "val_interval"),
    "early_stopping_patience": ("early_stopping_patience", "patience"),
}


_GENERIC_DEFAULTS = {
    "epochs": (100, 1000),
    "batch_size": (16,),
    "image_size": (640,),
    "optimizer": ("auto",),
    "learning_rate": (1e-2,),
    "weight_decay": (5e-4,),
    "seed": (0,),
    "log_interval": (20,),
    "warmup_epochs": (3.0,),
    "early_stopping_patience": (300,),
    "ema": (True,),
    "amp": (True,),
}


def _same_default(value: Any, defaults: Iterable[Any]) -> bool:
    for default in defaults:
        if isinstance(value, (int, float)) and isinstance(default, (int, float)):
            if abs(float(value) - float(default)) <= 1e-12:
                return True
        elif value == default:
            return True
    return False


def apply_animalvitpose_defaults(
    values: Dict[str, Any],
    model_scale: str = "small",
    model_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], list]:
    """Apply the MMPose recipe where the project still contains generic defaults.

    Non-default project settings remain authoritative, so users can still tune the shared
    training form for a particular experiment.
    """
    resolved = dict(values or {})
    recipe = dict(ANIMALVITPOSE_RECIPE)
    layer_decay_by_scale = recipe.pop("layer_decay_rate_by_scale", {}) or {}
    scale_name = normalize_animalvitpose_scale(model_scale)
    applied = []
    for key, recipe_value in recipe.items():
        aliases = _ALIASES.get(key, (key,))
        present = [(alias, resolved[alias]) for alias in aliases if alias in resolved]
        if not present:
            resolved[key] = recipe_value
            applied.append(key)
            continue

        markers = _GENERIC_DEFAULTS.get(key)
        is_default = markers is not None and all(
            _same_default(value, markers) for _alias, value in present
        )
        if is_default:
            resolved[key] = recipe_value
            applied.append(key)

    # Layer decay is an optimizer choice, so its per-size defaults stay in the
    # training profile instead of being mixed into the model architecture yaml.
    if resolved.get("layer_decay_rate") is None and scale_name in layer_decay_by_scale:
        resolved["layer_decay_rate"] = layer_decay_by_scale[scale_name]
        applied.append("layer_decay_rate")

    # The MMPose configs initialize the ViT backbone from the matching MAE checkpoint.
    # A user-provided local/remote checkpoint remains authoritative; ``pretrained: false``
    # explicitly disables the default initialization.
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


__all__ = [
    "ANIMALVITPOSE_RECIPE",
    "ANIMALVITPOSE_VARIANTS",
    "ANIMALVITPOSE_MODEL_DEFAULTS",
    "ANIMALVITPOSE_SIMCC",
    "ANIMALVITPOSE_PREPROCESSING",
    "ANIMALVITPOSE_AUGMENTATION",
    "ANIMALVITPOSE_VALIDATION",
    "KEYPOINT_METRICS",
    "ANIMALRTPOSE_PROFILE",
    "apply_animalvitpose_defaults",
    "configure_animalvitpose_model",
    "flatten_project_training_config",
    "materialize_project_training_config",
    "nested_other_config_from_flat",
    "normalize_animalvitpose_scale",
    "project_training_defaults",
    "training_profile_defaults",
]
