"""Three-layer configuration resolution: defaults < project < overrides.

The project's ``configs/runtime.yaml`` carries a section per scope
(``common`` / ``train`` / ``predict`` / ``export``). Resolution flattens each
layer to dotted paths (``export.opset``, ``predict.pose_filter.method``),
merges them in priority order and records where every final value came from.

Two deliberate choices, both from the refactor plan:

- **No legacy reads.** A project still carrying ``configs/other.yaml`` is
  refused with a message naming the new file, instead of being silently
  translated. (Author's decision, 2026-10-14: test-stage projects, no
  migration.)
- **Ownership discipline.** Keys in a project file that the registry does not
  know are refused, with a hint when the spelling looks like a known legacy
  name (``batch`` -> ``batch_size``). The train scope is exempt from the
  strict check for now: its section is a two-profile tree whose registered
  paths carry the active model's name, which the package-cleanup wave will
  pin down (see the refactor plan, wave W2).

``infer`` has no scope on purpose: it reads no project files.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from animalposetracker.config.defaults import default_subtree
from animalposetracker.config.schema import SCOPES, field_for, find_by_legacy

__all__ = [
    "ConfigFormatError",
    "ResolvedValue",
    "ResolvedScopes",
    "RUNTIME_CONFIG_RELATIVE",
    "project_dir_from",
    "read_project_yaml",
    "read_runtime_tree",
    "resolve_scopes",
]

RUNTIME_CONFIG_RELATIVE = "configs/runtime.yaml"

#: Scopes whose project-file keys are checked against the registry one by one.
_STRICT_SCOPES: Tuple[str, ...] = ("common", "predict", "export")


class ConfigFormatError(ValueError):
    """The project carries configuration this version cannot read."""


@dataclass(frozen=True)
class ResolvedValue:
    """One final value and where it came from."""

    value: Any
    source: str  # "default" | "project" | "cli"


@dataclass(frozen=True)
class ResolvedScopes:
    """The merged view over the requested scopes, keyed by dotted path."""

    values: Mapping[str, ResolvedValue]

    def value(self, path: str, default: Any = None) -> Any:
        entry = self.values.get(path)
        return entry.value if entry is not None else default

    def source(self, path: str) -> Optional[str]:
        entry = self.values.get(path)
        return entry.source if entry is not None else None

    def flat_for(self, scope: str) -> Dict[str, Any]:
        """The scope's plain values without the scope prefix or sources.

        This is what a flow builds its working ``values`` mapping from; the
        keys are the leaf names the flow's code already speaks.
        """
        prefix = f"{scope}."
        flat: Dict[str, Any] = {}
        for path, entry in self.values.items():
            if path.startswith(prefix):
                flat[path[len(prefix):]] = entry.value
        return flat

    def cli_overrides(self) -> Dict[str, Any]:
        """Only the values the command line re-decided (for the banner)."""
        return {
            path: entry.value
            for path, entry in self.values.items()
            if entry.source == "cli"
        }

    def project_values(self) -> Dict[str, Any]:
        """Only the values the project file re-decided."""
        return {
            path: entry.value
            for path, entry in self.values.items()
            if entry.source == "project"
        }


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def project_dir_from(config_arg) -> Path:
    """Accept a project directory or the path of its ``configs/runtime.yaml``."""
    path = Path(config_arg).expanduser().resolve()
    if path.is_dir():
        return path
    if path.name in ("runtime.yaml", "other.yaml") and path.parent.name == "configs":
        return path.parent.parent
    return path.parent  # a custom file name: the project is its directory's parent


def read_project_yaml(project_dir: Path, name: str, *, required: bool = True) -> Dict[str, Any]:
    """Read one YAML mapping from the project; empty when absent and optional."""
    path = Path(project_dir) / name
    if not path.is_file():
        if required:
            raise ConfigFormatError(f"Project file is missing: {path}")
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ConfigFormatError(f"Expected a YAML mapping in {path}")
    return value


def read_runtime_tree(project_dir: Path) -> Dict[str, Any]:
    """Read ``configs/runtime.yaml``; refuse anything older with a clear message."""
    project_dir = Path(project_dir)
    runtime_path = project_dir / RUNTIME_CONFIG_RELATIVE
    if not runtime_path.is_file():
        legacy_path = project_dir / "configs" / "other.yaml"
        if legacy_path.is_file():
            raise ConfigFormatError(
                f"Legacy configuration detected ({legacy_path}). This version reads "
                f"{RUNTIME_CONFIG_RELATIVE} only; the old format is no longer supported. "
                "Rebuild the project (test-stage projects do not need migrating)."
            )
        raise ConfigFormatError(f"Project is missing {runtime_path}")

    with runtime_path.open("r", encoding="utf-8") as stream:
        tree = yaml.safe_load(stream) or {}
    if not isinstance(tree, dict):
        raise ConfigFormatError(f"Expected a YAML mapping in {runtime_path}")

    unknown = [str(key) for key in tree if key not in SCOPES]
    if unknown:
        raise ConfigFormatError(
            f"{runtime_path} carries unknown sections {', '.join(sorted(unknown))}; "
            f"expected only {', '.join(SCOPES)}"
        )
    return tree


# ---------------------------------------------------------------------------
# flattening and validation
# ---------------------------------------------------------------------------

def _flatten(tree: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested mapping to dotted paths; scalars (incl. null) are leaves."""
    flat: Dict[str, Any] = {}
    for key, value in (tree or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _check_project_keys(scope: str, subtree: Mapping[str, Any], source_name: str) -> None:
    """Refuse keys the registry does not own (strict scopes only)."""
    if scope not in _STRICT_SCOPES:
        return
    for path in _flatten(subtree):
        key = path.rsplit(".", 1)[-1]
        group = path.rsplit(".", 1)[0] if "." in path else ""
        if field_for(key, scope, group) is not None:
            continue
        hint = ""
        for candidate in find_by_legacy(key):
            if candidate.scope == scope:
                hint = f" (did you mean '{candidate.key}'? the old spelling '{key}' is not read)"
                break
        raise ConfigFormatError(
            f"Unknown configuration key '{scope}.{path}' in {source_name}{hint}"
        )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def resolve_scopes(
    project_dir,
    scopes: Sequence[str],
    overrides: Optional[Mapping[str, Any]] = None,
) -> ResolvedScopes:
    """Merge the three layers for the requested scopes.

    Args:
        project_dir: The project directory (or its ``configs/runtime.yaml``).
        scopes: Scope names to resolve, e.g. ``("common", "export")``.
        overrides: Explicit run-time values keyed by dotted path, highest
            priority (``{"export.opset": 18}``).
    """
    project_dir = Path(project_dir)
    for scope in scopes:
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected one of {SCOPES}")

    project_tree = read_runtime_tree(project_dir)
    source_name = str(project_dir / RUNTIME_CONFIG_RELATIVE)

    merged: Dict[str, ResolvedValue] = {}
    for scope in scopes:
        defaults = default_subtree(scope)
        for path, value in _flatten(defaults).items():
            merged[f"{scope}.{path}"] = ResolvedValue(value, "default")

    for scope in scopes:
        subtree = project_tree.get(scope) or {}
        if not isinstance(subtree, dict):
            raise ConfigFormatError(f"Section '{scope}' in {source_name} must be a mapping")
        _check_project_keys(scope, subtree, source_name)
        for path, value in _flatten(subtree).items():
            merged[f"{scope}.{path}"] = ResolvedValue(deepcopy(value), "project")

    for path, value in (overrides or {}).items():
        scope = str(path).split(".", 1)[0]
        if scope not in SCOPES:
            raise ValueError(
                f"Override path {path!r} starts with unknown scope {scope!r}; "
                f"expected one of {SCOPES}"
            )
        if scope not in scopes:
            continue  # an override for a scope we were not asked to resolve
        _check_override_path(str(path))
        merged[str(path)] = ResolvedValue(value, "cli")

    return ResolvedScopes(values=merged)


def _check_override_path(path: str) -> None:
    """Overrides come from our own CLI code; a wrong path is a coding bug."""
    scope, _, rest = path.partition(".")
    key = rest.rsplit(".", 1)[-1]
    group = rest.rsplit(".", 1)[0] if "." in rest else ""
    if scope in _STRICT_SCOPES and field_for(key, scope, group) is None:
        raise ValueError(f"Override path {path!r} is not a registered configuration key")
