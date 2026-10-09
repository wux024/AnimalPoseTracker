"""AnimalViTPose data-pipeline defaults sourced from the built-in YAML profiles."""

from pathlib import Path

import yaml

from animalposetracker.cfg import MODEL_YAML_PATHS, TRAINING_CFG_PATH

with Path(TRAINING_CFG_PATH).open("r", encoding="utf-8") as stream:
    _training_tree = yaml.safe_load(stream) or {}
with Path(MODEL_YAML_PATHS["AnimalViTPose"]).open("r", encoding="utf-8") as stream:
    _model_defaults = yaml.safe_load(stream) or {}

_vitpose_profile = (_training_tree.get("models", {}) or {}).get("AnimalViTPose", {}) or {}
ANIMALVITPOSE_AUGMENTATION = dict(_vitpose_profile.get("augmentation", {}) or {})
ANIMALVITPOSE_PREPROCESSING = dict(_model_defaults.get("preprocessing", {}) or {})

__all__ = ["ANIMALVITPOSE_AUGMENTATION", "ANIMALVITPOSE_PREPROCESSING"]
