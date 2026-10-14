"""The resolver merges three layers and records where values came from.

Also pins the writer's round trip and the new-format context loader that
``export`` now uses -- the first flow migrated off the legacy other.yaml
path (refactor wave W1).
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from animalposetracker.config import (
    ConfigFormatError,
    project_dir_from,
    resolve_scopes,
    write_scope,
)
from animalposetracker.project.model_context import load_project_runtime_context


PROJECT_YAML = {
    "project_name": "probe",
    "worker": "wux",
    "model_type": "AnimalRTPose",
    "model_scale": "N",
    "date": "20261014",
}

DATASET_YAML = {
    "names": {0: "probe"},
    "kpt_shape": [2, 3],
}

MODEL_YAML = {
    "nc": 1,
    "kpt_shape": [2, 3],
}


def _make_project(temporary: str, runtime: dict | None = None, legacy: bool = False) -> Path:
    root = Path(temporary)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "project.yaml").write_text(yaml.safe_dump(PROJECT_YAML), encoding="utf-8")
    name = "other.yaml" if legacy else "runtime.yaml"
    (root / "configs" / name).write_text(yaml.safe_dump(runtime or {}), encoding="utf-8")
    (root / "configs" / "dataset.yaml").write_text(yaml.safe_dump(DATASET_YAML), encoding="utf-8")
    (root / "configs" / "model.yaml").write_text(yaml.safe_dump(MODEL_YAML), encoding="utf-8")
    return root


class ResolverTests(unittest.TestCase):
    def test_defaults_fill_unspecified_keys(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            resolved = resolve_scopes(root, ("common", "predict", "export"))

        self.assertEqual(resolved.value("export.opset"), 17)
        self.assertEqual(resolved.source("export.opset"), "default")
        self.assertEqual(resolved.value("predict.confidence"), 0.25)
        self.assertEqual(resolved.value("common.device"), "auto")
        self.assertIsNone(resolved.value("export.format"))

    def test_project_values_beat_defaults(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"export": {"format": "onnx", "opset": 16}})
            resolved = resolve_scopes(root, ("export",))

        self.assertEqual(resolved.value("export.format"), "onnx")
        self.assertEqual(resolved.source("export.format"), "project")
        self.assertEqual(resolved.value("export.opset"), 16)
        # Untouched keys still come from the defaults.
        self.assertEqual(resolved.value("export.workspace"), 4.0)

    def test_cli_overrides_beat_everything(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"export": {"opset": 16, "simplify": False}})
            resolved = resolve_scopes(root, ("export",), overrides={"export.opset": 14})

        self.assertEqual(resolved.value("export.opset"), 14)
        self.assertEqual(resolved.source("export.opset"), "cli")
        self.assertEqual(resolved.cli_overrides(), {"export.opset": 14})
        # Only the final provenance is recorded: the shadowed project value
        # does not appear, while untouched project keys still do.
        self.assertEqual(resolved.project_values(), {"export.simplify": False})

    def test_partial_project_section_merges_deeply(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"predict": {"pose_filter": {"method": "savgol"}}})
            resolved = resolve_scopes(root, ("predict",))

        self.assertEqual(resolved.value("predict.pose_filter.method"), "savgol")
        # The sibling keys of the same mapping block keep their defaults.
        self.assertEqual(resolved.value("predict.pose_filter.window_length"), 5)
        self.assertEqual(resolved.value("predict.confidence"), 0.25)

    def test_legacy_project_is_refused_by_name(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"application": {"format": "onnx"}}, legacy=True)
            with self.assertRaises(ConfigFormatError) as raised:
                resolve_scopes(root, ("export",))

        self.assertIn("runtime.yaml", str(raised.exception))
        self.assertIn("other.yaml", str(raised.exception))

    def test_missing_runtime_file_is_refused(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            (root / "configs" / "runtime.yaml").unlink()
            with self.assertRaises(ConfigFormatError):
                resolve_scopes(root, ("export",))

    def test_unknown_key_is_refused_with_legacy_hint(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"predict": {"prediction_confidence": 0.4}})
            with self.assertRaises(ConfigFormatError) as raised:
                resolve_scopes(root, ("predict",))

        message = str(raised.exception)
        self.assertIn("prediction_confidence", message)
        self.assertIn("confidence", message)  # the hint names the canonical key

    def test_unknown_section_is_refused(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"infer": {"confidence": 0.3}})
            with self.assertRaises(ConfigFormatError):
                resolve_scopes(root, ("export",))

    def test_overrides_for_unresolved_scopes_are_skipped(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            resolved = resolve_scopes(root, ("export",), overrides={"predict.iou": 0.5})

        self.assertIsNone(resolved.value("predict.iou"))

    def test_override_paths_are_checked(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            with self.assertRaises(ValueError):
                resolve_scopes(root, ("export",), overrides={"exports.opset": 1})
            with self.assertRaises(ValueError):
                resolve_scopes(root, ("export",), overrides={"export.op_set": 1})

    def test_flat_for_strips_the_scope_prefix(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"export": {"format": "onnx"}})
            resolved = resolve_scopes(root, ("common", "export"))

        export_flat = resolved.flat_for("export")
        self.assertEqual(export_flat["format"], "onnx")
        self.assertEqual(export_flat["workspace"], 4.0)
        self.assertNotIn("device", export_flat)  # that one lives in common
        self.assertEqual(resolved.flat_for("common")["device"], "auto")

    def test_nested_for_rebuilds_mapping_blocks(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"predict": {"pose_filter": {"method": "savgol"}}})
            resolved = resolve_scopes(root, ("predict",))

        predict_tree = resolved.nested_for("predict")
        self.assertEqual(predict_tree["pose_filter"]["method"], "savgol")
        self.assertEqual(predict_tree["pose_filter"]["window_length"], 5)
        self.assertIsInstance(predict_tree["tracking"], dict)
        self.assertFalse(predict_tree["tracking"]["enabled"])
        self.assertEqual(predict_tree["confidence"], 0.25)

    def test_changed_from_defaults_ignores_snapshot_values(self):
        # A project snapshot carries every registered key; only values that
        # actually differ from the package defaults count as changed.
        with TemporaryDirectory() as temporary:
            root = _make_project(
                temporary,
                {"export": {"format": "onnx", "opset": 17}},  # 17 is the default
            )
            resolved = resolve_scopes(root, ("export",))

        self.assertEqual(resolved.changed_from_defaults(), {"export.format": "onnx"})

    def test_train_profiles_merge_per_model(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(
                temporary,
                {"train": {"models": {"AnimalViTPose": {"training": {"epochs": 300}}}}},
            )
            resolved = resolve_scopes(root, ("train",))

        self.assertEqual(
            resolved.value("train.models.AnimalRTPose.training.epochs"), 1000
        )
        self.assertEqual(
            resolved.value("train.models.AnimalViTPose.training.epochs"), 300
        )
        self.assertEqual(resolved.source("train.models.AnimalViTPose.training.epochs"), "project")
        # The other model's profile is untouched by the override.
        self.assertEqual(
            resolved.source("train.models.AnimalRTPose.training.epochs"), "default"
        )

    def test_project_dir_accepts_directory_or_file(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            self.assertEqual(project_dir_from(root), root.resolve())
            self.assertEqual(project_dir_from(root / "configs" / "runtime.yaml"), root.resolve())


class WriterTests(unittest.TestCase):
    def test_roundtrip_updates_one_scope(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"export": {"format": "onnx"}, "predict": {"iou": 0.5}})
            write_scope(root, "export", {"format": "engine", "workspace": 8.0})

            resolved = resolve_scopes(root, ("predict", "export"))

        self.assertEqual(resolved.value("export.format"), "engine")
        self.assertEqual(resolved.value("export.workspace"), 8.0)
        self.assertEqual(resolved.value("export.opset"), 17)  # not written back: default applies
        self.assertEqual(resolved.value("predict.iou"), 0.5)  # the other section survived

    def test_writer_refuses_legacy_projects(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {}, legacy=True)
            with self.assertRaises(ConfigFormatError):
                write_scope(root, "export", {"format": "onnx"})


class RuntimeContextTests(unittest.TestCase):
    def test_context_assembles_from_runtime_yaml(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary, {"export": {"format": "onnx"}})
            config, context = load_project_runtime_context(root)

        self.assertEqual(config.image_size, 640)  # AnimalRTPose profile default
        self.assertEqual(config.device, "auto")
        self.assertEqual(Path(config.model), root / "configs" / "model.yaml")
        self.assertEqual(Path(config.data), root / "configs" / "dataset.yaml")
        self.assertEqual(context["runtime"]["format"], "onnx")
        self.assertEqual(context["runtime"]["device"], "auto")
        self.assertEqual(context["resolved"].source("export.format"), "project")

    def test_context_follows_the_active_model_profile(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            (root / "project.yaml").write_text(
                yaml.safe_dump({**PROJECT_YAML, "model_type": "AnimalViTPose"}),
                encoding="utf-8",
            )
            config, _context = load_project_runtime_context(root)

        self.assertEqual(config.image_size, 256)  # the ViTPose profile's size

    def test_missing_model_file_is_reported(self):
        with TemporaryDirectory() as temporary:
            root = _make_project(temporary)
            (root / "configs" / "model.yaml").unlink()
            with self.assertRaises(FileNotFoundError):
                load_project_runtime_context(root)


if __name__ == "__main__":
    unittest.main()
