"""Configuration layer for AnimalPoseTracker (refactor wave W1).

What exists today: the *definition layer* (the key registry in ``schema`` and
the defaults access in ``defaults``), the *resolver* (three-layer merge with
source tracking) and the *writer* (the one write path for ``runtime.yaml``).
Flows migrate onto this package one by one; ``export`` is the first.

Boundary (from the refactor plan): this package serves the project-side flows
(train / predict / export / GUI). The ``inference`` pipeline must keep its
dependency closure free of this package -- it reads only CLI arguments and
the artifact sidecar.
"""

from animalposetracker.config.defaults import (
    RUNTIME_DEFAULTS_PATH,
    default_subtree,
    missing_field_defaults,
    orphan_default_paths,
    runtime_defaults,
    training_defaults,
)
from animalposetracker.config.resolver import (
    RUNTIME_CONFIG_RELATIVE,
    ConfigFormatError,
    ResolvedScopes,
    ResolvedValue,
    project_dir_from,
    read_project_yaml,
    read_runtime_tree,
    resolve_scopes,
)
from animalposetracker.config.schema import (
    APPLY_MODES,
    FIELDS,
    SCOPES,
    TRAIN_GROUPS,
    VISIBILITIES,
    Field,
    display_path,
    field_for,
    fields,
    find_by_legacy,
    to_cli_name,
)
from animalposetracker.config.writer import write_scope

__all__ = [
    # schema
    "Field",
    "FIELDS",
    "SCOPES",
    "APPLY_MODES",
    "VISIBILITIES",
    "TRAIN_GROUPS",
    "fields",
    "field_for",
    "to_cli_name",
    "display_path",
    "find_by_legacy",
    # defaults
    "RUNTIME_DEFAULTS_PATH",
    "runtime_defaults",
    "training_defaults",
    "default_subtree",
    "missing_field_defaults",
    "orphan_default_paths",
    # resolver
    "RUNTIME_CONFIG_RELATIVE",
    "ConfigFormatError",
    "ResolvedScopes",
    "ResolvedValue",
    "project_dir_from",
    "read_project_yaml",
    "read_runtime_tree",
    "resolve_scopes",
    # writer
    "write_scope",
]
