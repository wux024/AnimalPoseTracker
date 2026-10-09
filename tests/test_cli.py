import unittest
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import yaml

from animalposetracker.cli import main


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


if __name__ == "__main__":
    unittest.main()
