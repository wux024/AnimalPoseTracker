"""Default values: the read side of the defaults files.

Values live in exactly two package files. ``cfg/training.yaml`` carries the
``train`` scope (existing file, unchanged here); ``cfg/runtime_defaults.yaml``
carries ``common`` / ``predict`` / ``export``. This module is the one access
point -- callers ask it for a scope's default subtree instead of reading the
YAML themselves, so "where does the default come from" has a single answer.

The ``infer`` pipeline is deliberately absent: it reads no project files and
its defaults live in the neutral constants (``postprocessing/nms_params.py``
and friends), shared with this package rather than owned by it.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

from animalposetracker.cfg import TRAINING_CFG_PATH

__all__ = [
    "RUNTIME_DEFAULTS_PATH",
    "MODEL_SEED_DEFAULTS",
    "runtime_defaults",
    "training_defaults",
    "default_subtree",
    "missing_field_defaults",
    "orphan_default_paths",
]

RUNTIME_DEFAULTS_PATH = Path(TRAINING_CFG_PATH).parent / "runtime_defaults.yaml"

#: Recipes whose own training setup seeds differently. `create` writes the value into the one
#: `train.shared.runtime.seed` key, so a project has a single seed to read and edit.
MODEL_SEED_DEFAULTS: Dict[str, int] = {"AnimalViTPose": 21}

_RUNTIME_SCOPES: Tuple[str, ...] = ("common", "predict", "export")


@lru_cache(maxsize=None)
def _read(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def runtime_defaults() -> Mapping[str, Any]:
    """The whole runtime defaults tree (common / predict / export)."""
    return _read(RUNTIME_DEFAULTS_PATH)


def training_defaults() -> Mapping[str, Any]:
    """The whole training defaults tree (shared / metrics / models)."""
    return _read(TRAINING_CFG_PATH)


def default_subtree(scope: str) -> Dict[str, Any]:
    """A deep copy of one scope's default subtree.

    ``train`` returns the whole training.yaml tree; the runtime scopes return
    their section of runtime_defaults.yaml (empty mapping when absent).
    """
    if scope == "train":
        return deepcopy(dict(training_defaults()))
    if scope not in _RUNTIME_SCOPES:
        raise ValueError(
            f"Unknown scope {scope!r}; expected one of {('train',) + _RUNTIME_SCOPES}"
        )
    return deepcopy(dict(runtime_defaults().get(scope) or {}))


def _flatten(scope: str) -> List[Tuple[str, str, str]]:
    """The (scope, group, key) triples present in a runtime defaults section.

    A mapping value becomes a group (``tracking.enabled``); scalars (including
    null) become plain keys.
    """
    entries: List[Tuple[str, str, str]] = []
    for key, value in (runtime_defaults().get(scope) or {}).items():
        if isinstance(value, dict):
            for sub_key in value:
                entries.append((scope, str(key), str(sub_key)))
        else:
            entries.append((scope, "", str(key)))
    return entries


def missing_field_defaults() -> List[str]:
    """Registered common/predict/export fields that have no default value.

    Empty means the defaults file covers every runtime field in the schema.
    (The train scope's alignment is checked when training.yaml adopts the
    canonical spellings, in the package-cleanup wave.)
    """
    from animalposetracker.config.schema import fields

    present = set()
    for scope in _RUNTIME_SCOPES:
        present.update(_flatten(scope))

    missing = []
    for field in fields():
        if field.scope in _RUNTIME_SCOPES and (field.scope, field.group, field.key) not in present:
            missing.append(field.path)
    return missing


def orphan_default_paths() -> List[str]:
    """Paths in the runtime defaults file that no registered field owns."""
    from animalposetracker.config.schema import fields

    known = {(field.scope, field.group, field.key) for field in fields()}
    orphans = []
    for scope in _RUNTIME_SCOPES:
        for entry in _flatten(scope):
            if entry not in known:
                orphans.append(".".join(part for part in entry if part))
    return orphans
