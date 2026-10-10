import unittest
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import yaml

from animalposetracker.cli import main
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.training.cli import _resolve_project_default_pretrained


class CommandDispatcherTests(unittest.TestCase):
    def test_train_forwards_arguments(self):
        with patch("animalposetracker.training.cli.run", return_value=0) as run:
            self.assertEqual(main(["train", "--config", "project/configs/other.yaml"]), 0)
        run.assert_called_once_with(["--config", "project/configs/other.yaml"])

    def test_val_selects_validation_only_mode(self):
        with patch("animalposetracker.training.cli.run", return_value=0) as run:
            self.assertEqual(main(["val", "--config", "project/configs/other.yaml", "--weights", "best.pt"]), 0)
        run.assert_called_once_with([
            "--validate-only", "--config", "project/configs/other.yaml", "--weights", "best.pt",
        ])

    def test_val_help_is_specific_to_validation(self):
        with patch("sys.stdout") as output:
            with self.assertRaises(SystemExit) as raised:
                main(["val", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_predict_and_export_dispatch(self):
        prediction_cli = ModuleType("animalposetracker.prediction.cli")
        export_cli = ModuleType("animalposetracker.export.cli")
        prediction_cli.run = Mock(return_value=7)
        export_cli.run = Mock(return_value=0)
        with patch.dict("sys.modules", {
            "animalposetracker.prediction.cli": prediction_cli,
            "animalposetracker.export.cli": export_cli,
        }):
            self.assertEqual(main(["predict", "--weights", "best.pt"]), 7)
            prediction_cli.run.assert_called_once_with(["--weights", "best.pt"])

            self.assertEqual(main(["export", "--weights", "best.pt", "--format", "onnx"]), 0)
            export_cli.run.assert_called_once_with(["--weights", "best.pt", "--format", "onnx"])

    def test_project_create_writes_project_from_dataset_yaml(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_yaml = root / "trimouse.yaml"
            dataset_yaml.write_text(yaml.safe_dump({
                "path": "unused",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [2, 3],
                "names": {0: "mouse"},
                "skeleton": [[0, 1]],
            }), encoding="utf-8")
            workspace = root / "workspace"
            dataset_root = root / "dataset"

            with patch("sys.stdout"):
                result = main([
                    "create",
                    "--dataset", str(dataset_yaml),
                    "--dataset-root", str(dataset_root),
                    "--workspace", str(workspace),
                    "--name", "trimouse",
                    "--worker", "test",
                    "--model", "AnimalRTPose",
                    "--scale", "N",
                    "--date", "20261009-120000",
                    "--no-pretrained",
                ])

            project = workspace / "trimouse-test-AnimalRTPose-N-20261009-120000"
            self.assertEqual(result, 0)
            self.assertTrue((project / "project.yaml").is_file())
            self.assertTrue((project / "configs" / "model.yaml").is_file())
            self.assertTrue((project / "configs" / "other.yaml").is_file())
            saved_data = yaml.safe_load((project / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_data["path"], str(dataset_root.resolve()))
            self.assertEqual(saved_data["annotation_format"], "yolo")
            saved_other = yaml.safe_load((project / "configs" / "other.yaml").read_text(encoding="utf-8"))
            self.assertFalse(
                saved_other["training"]["shared"]["runtime"]["pretrained"]
            )

    def test_no_arguments_prints_help(self):
        with patch("sys.stdout"):
            self.assertEqual(main([]), 0)


def _write_minimal_dataset_yaml(root: Path) -> Path:
    dataset_yaml = root / "trimouse.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": "unused",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [2, 3],
        "names": {0: "mouse"},
        "skeleton": [[0, 1]],
    }), encoding="utf-8")
    return dataset_yaml


def _run_create(root: Path, *extra_args):
    dataset_yaml = _write_minimal_dataset_yaml(root)
    workspace = root / "workspace"
    dataset_root = root / "dataset"
    argv = [
        "create",
        "--dataset", str(dataset_yaml),
        "--dataset-root", str(dataset_root),
        "--workspace", str(workspace),
        "--name", "trimouse",
        "--worker", "test",
        "--model", "AnimalRTPose",
        "--scale", "N",
        "--date", "20261009-120000",
        *extra_args,
    ]
    with patch("sys.stdout"):
        result = main(argv)
    project = workspace / "trimouse-test-AnimalRTPose-N-20261009-120000"
    saved_other = yaml.safe_load((project / "configs" / "other.yaml").read_text(encoding="utf-8"))
    return result, saved_other


class ProjectCreatePretrainedTests(unittest.TestCase):
    def test_create_with_default_pretrained_does_not_detect_or_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(
                AnimalPoseTrackerProject, "_detect_pretrained",
                side_effect=AssertionError("create must not resolve pretrained weights"),
            ):
                result, saved_other = _run_create(root)
            self.assertEqual(result, 0)
            self.assertTrue(saved_other["training"]["shared"]["runtime"]["pretrained"])

    def test_create_records_explicit_pretrained_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weights_path = root / "weights" / "animalrtpose-n.pt"
            result, saved_other = _run_create(root, "--pretrained", str(weights_path))
            self.assertEqual(result, 0)
            self.assertEqual(
                saved_other["training"]["shared"]["runtime"]["pretrained"],
                str(weights_path),
            )

    def test_create_no_pretrained_flag_still_records_false(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result, saved_other = _run_create(Path(temporary_directory), "--no-pretrained")
            self.assertEqual(result, 0)
            self.assertFalse(saved_other["training"]["shared"]["runtime"]["pretrained"])


class DefaultPretrainedResolutionTests(unittest.TestCase):
    def test_missing_checkpoint_raises_with_guidance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            project_values = {"model_type": "AnimalRTPose", "model_scale": "N"}
            with self.assertRaises(FileNotFoundError) as raised:
                _resolve_project_default_pretrained(project_dir, project_values)
            message = str(raised.exception)
            self.assertIn("pretrained", message)
            self.assertIn("animalrtpose-n.pt", message)

    def test_existing_checkpoint_is_resolved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            checkpoint = project_dir / "pretrained" / "animalrtpose-n.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"weights")
            resolved = _resolve_project_default_pretrained(
                project_dir, {"model_type": "AnimalRTPose", "model_scale": "N"}
            )
            self.assertEqual(resolved, checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
