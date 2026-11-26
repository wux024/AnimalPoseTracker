"""`--exist-ok` decides whether an existing export directory may be reused.

It used to be pushed through the configuration override table as ``export.exist_ok``,
a key the schema does not define, so passing the flag raised before any export ran.
"""

from pathlib import Path

import pytest

from animalposetracker.export import cli as export_cli


def test_exist_ok_is_not_a_config_override(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "runtime.yaml").write_text("project:\n  name: demo\n", encoding="utf-8")

    captured = {}

    def fake_resolve_scopes(project_dir, scopes, overrides=None):
        captured["overrides"] = dict(overrides or {})
        raise RuntimeError("stop after resolver")

    monkeypatch.setattr(export_cli, "resolve_scopes", fake_resolve_scopes)

    with pytest.raises(RuntimeError, match="stop after resolver"):
        export_cli.run(["--config", str(project), "--weights", "weights/best.pt", "--exist-ok"])

    assert "export.exist_ok" not in captured["overrides"]
    # The model options still travel through the resolver.
    assert captured["overrides"] == {}


def test_model_options_still_become_overrides(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "runtime.yaml").write_text("project:\n  name: demo\n", encoding="utf-8")

    captured = {}

    def fake_resolve_scopes(project_dir, scopes, overrides=None):
        captured["overrides"] = dict(overrides or {})
        raise RuntimeError("stop after resolver")

    monkeypatch.setattr(export_cli, "resolve_scopes", fake_resolve_scopes)

    with pytest.raises(RuntimeError, match="stop after resolver"):
        export_cli.run(
            ["--config", str(project), "--weights", "weights/best.pt", "--opset", "17", "--half"]
        )

    assert captured["overrides"]["export.opset"] == 17
    assert captured["overrides"]["export.half"] is True


def test_parser_treats_exist_ok_as_a_bare_switch():
    parser = export_cli._make_argument_parser()

    assert parser.parse_args(["--weights", "x.pt", "--exist-ok"]).exist_ok is True
    assert parser.parse_args(["--weights", "x.pt"]).exist_ok is None
