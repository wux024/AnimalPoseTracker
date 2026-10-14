"""The one write path for project configuration.

Everything that saves configuration -- the GUI forms, the CLI, future tools --
goes through here, so the new format has exactly one author. The writer
updates one scope at a time inside ``configs/runtime.yaml`` and leaves the
file's other sections untouched.

What this writer deliberately does not do: read or upgrade legacy files. A
project without ``runtime.yaml`` gets the same refusal the resolver gives.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from animalposetracker.config.resolver import (
    RUNTIME_CONFIG_RELATIVE,
    ConfigFormatError,
    read_runtime_tree,
)
from animalposetracker.config.schema import SCOPES

__all__ = ["write_scope"]


def write_scope(project_dir, scope: str, subtree: Mapping[str, Any]) -> Path:
    """Write one scope's subtree into ``configs/runtime.yaml``.

    Args:
        project_dir: The project directory.
        scope: One of the known scopes; other sections of the file stay as
            they are.
        subtree: The scope's full subtree (what gets written verbatim).

    Returns the path written.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; expected one of {SCOPES}")
    if not isinstance(subtree, Mapping):
        raise ValueError("The scope subtree must be a mapping")

    project_dir = Path(project_dir)
    tree = read_runtime_tree(project_dir)  # also enforces the legacy refusal
    tree[scope] = deepcopy(dict(subtree))

    runtime_path = project_dir / RUNTIME_CONFIG_RELATIVE
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with runtime_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                tree,
                stream,
                indent=2,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
    except OSError as error:
        raise ConfigFormatError(f"Failed to write {runtime_path}: {error}") from error
    return runtime_path
