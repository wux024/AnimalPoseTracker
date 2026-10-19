"""Render the configuration reference from the registry and the defaults.

The reference in ``docs/pytorch/pytorch_config.md`` is generated, not hand-written:
one key can be renamed in ``schema.py`` and the documentation follows. Run

    python -m animalposetracker.config.docs --write docs/pytorch/pytorch_config.md

to refresh the marked section in place, or without ``--write`` to print it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from animalposetracker.config.defaults import default_subtree
from animalposetracker.config.schema import FIELDS, SCOPES, Field

__all__ = ["render_reference", "write_reference", "BEGIN_MARKER", "END_MARKER"]

BEGIN_MARKER = "<!-- BEGIN GENERATED REFERENCE -->"
END_MARKER = "<!-- END GENERATED REFERENCE -->"


_MISSING = object()


def _type_name(field: Field) -> str:
    return " / ".join(getattr(t, "__name__", str(t)) for t in field.type)


def _lookup(tree: Mapping[str, Any], parts: Tuple[str, ...]) -> Any:
    node: Any = tree
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _render_value(value: Any) -> str:
    if value is _MISSING:
        return "—"
    if value is None:
        return "`null`"
    if isinstance(value, str):
        return f"`{value}`"
    if isinstance(value, list):
        return f"`{value}`"
    return str(value)


def _default_cell(field: Field, defaults) -> str:
    if field.scope == "train" and field.group.startswith("models."):
        section = field.group.split(".", 1)[1]
        models = defaults.get("models") or {}
        cells = [
            _render_value(
                _lookup(models.get(model) or {}, tuple(section.split(".")) + (field.key,))
            )
            for model in ("AnimalRTPose", "AnimalViTPose")
        ]
        return f"{cells[0]} / {cells[1]} (RTPose / ViTPose)"
    parts = tuple(field.group.split(".")) + (field.key,) if field.group else (field.key,)
    return _render_value(_lookup(defaults, parts))


def render_reference() -> str:
    """The generated reference section (markdown, markers not included)."""
    lines: List[str] = [
        "This reference is generated from the key registry "
        "(`animalposetracker/config/schema.py`) and the defaults files "
        "(`cfg/training.yaml`, `cfg/runtime_defaults.yaml`). Do not edit it by "
        "hand -- run `python -m animalposetracker.config.docs --write "
        "docs/pytorch/pytorch_config.md` after changing the registry.",
        "",
    ]
    for scope in SCOPES:
        scope_fields = [field for field in FIELDS if field.scope == scope]
        if not scope_fields:
            continue
        defaults = default_subtree(scope)
        lines.append(f"### Scope `{scope}`")
        lines.append("")
        seen_groups: List[str] = []
        for field in scope_fields:
            if field.group not in seen_groups:
                seen_groups.append(field.group)
        for group in seen_groups:
            title = f"{scope}.{group}" if group else scope
            lines.append(f"**`{title}`**")
            lines.append("")
            lines.append("| key | type | default | apply | visibility | notes |")
            lines.append("|---|---|---|---|---|---|")
            for field in scope_fields:
                if field.group != group:
                    continue
                lines.append(
                    f"| `{field.key}` | {_type_name(field)} | {_default_cell(field, defaults)} "
                    f"| {field.apply} | {field.visibility} | {field.note} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reference(path) -> Path:
    """Replace the marked section of ``path`` with a fresh rendering."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; create it with the BEGIN/END markers first"
        )
    text = path.read_text(encoding="utf-8")
    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if begin == -1 or end == -1 or end < begin:
        raise ValueError(f"{path} carries no {BEGIN_MARKER} / {END_MARKER} section")
    rendered = render_reference()
    updated = (
        text[: begin + len(BEGIN_MARKER)]
        + "\n\n"
        + rendered
        + "\n"
        + text[end:]
    )
    path.write_text(updated, encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(render_reference(), end="")
        return 0
    if arguments[0] == "--write" and len(arguments) == 2:
        written = write_reference(arguments[1])
        print(f"Reference updated: {written}")
        return 0
    print("usage: python -m animalposetracker.config.docs [--write docs/pytorch/pytorch_config.md]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
