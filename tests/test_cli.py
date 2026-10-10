import io
import unittest
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from animalposetracker.cli import main
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.training.cli import _resolve_project_default_pretrained
from animalposetracker.training.console import PrettyTrainingRenderer
from animalposetracker.training.events import TrainingEvent


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

    def test_infer_dispatch(self):
        inference_cli = ModuleType("animalposetracker.inference.cli")
        inference_cli.run = Mock(return_value=3)
        with patch.dict("sys.modules", {"animalposetracker.inference.cli": inference_cli}):
            self.assertEqual(main(["infer", "--weights", "m.onnx", "--source", "0"]), 3)
            inference_cli.run.assert_called_once_with(["--weights", "m.onnx", "--source", "0"])

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


class PrettyTrainingRendererTests(unittest.TestCase):
    def _render(self, events):
        stream = io.StringIO()
        renderer = PrettyTrainingRenderer(stream=stream)
        for event in events:
            renderer.emit(event)
        return stream.getvalue()

    def test_batch_events_render_in_place_with_progress(self):
        output = self._render([
            TrainingEvent(
                event="batch", epoch=1, epochs=2, step=1, steps=9,
                metrics={"loss": 22.5, "box": 0.41, "pose": 0.9},
            ),
        ])
        self.assertTrue(output.startswith("\r"))
        self.assertIn("epoch 1/2", output)
        self.assertIn("step 1/9", output)
        self.assertIn("loss=22.5", output)

    def test_known_events_render_compact_lines(self):
        output = self._render([
            TrainingEvent(
                event="pretrained",
                metrics={"loaded_tensors": 423.0, "shape_mismatches": 42.0},
            ),
            TrainingEvent(event="started", epoch=0, epochs=2, message="device=cuda"),
            TrainingEvent(
                event="validation", epoch=1, epochs=2,
                metrics={"loss": 20.6, "coco/AP": 0.0, "PCK": 0.0},
            ),
            TrainingEvent(
                event="epoch_end", epoch=1, epochs=2,
                metrics={"train_loss": 22.4, "val_loss": 20.6, "val_fitness": 0.0},
            ),
            TrainingEvent(event="finished", epoch=2, epochs=2),
        ])
        self.assertIn("[pretrained] loaded 423 tensors (42 incompatible shapes skipped)", output)
        self.assertIn("[started] device=cuda", output)
        self.assertIn("[val] epoch 1/2", output)
        self.assertIn("[epoch 1/2] train_loss=22.4 val_loss=20.6", output)
        self.assertIn("[finished]", output)
        self.assertNotIn('"event"', output)

    def test_unknown_event_falls_back_to_tagged_line(self):
        output = self._render([
            TrainingEvent(event="custom", message="hello", metrics={"x": 1.5}),
        ])
        self.assertIn("[custom] hello x=1.5", output)


class InferCliTests(unittest.TestCase):
    def test_resolve_num_classes_from_runtime_shape(self):
        from animalposetracker.inference.cli import _resolve_num_classes

        engine = SimpleNamespace(_runtime_output_shapes={"predictions": (1, 43, 8400)})
        self.assertEqual(_resolve_num_classes(engine, (12, 3)), 3)
        empty = SimpleNamespace(_runtime_output_shapes={})
        self.assertEqual(_resolve_num_classes(empty, (12, 3)), 1)

    def test_infer_parser_defaults(self):
        from animalposetracker.inference.cli import _make_argument_parser

        args = _make_argument_parser().parse_args(["--weights", "m.onnx", "--source", "0"])
        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.iou, 0.45)
        self.assertEqual(args.pose_filter, "none")
        self.assertFalse(args.show)
        self.assertIsNone(args.tracker)


class ConsoleBannerTests(unittest.TestCase):
    def test_environment_line_mentions_package_and_python(self):
        from animalposetracker.training.console import environment_line

        line = environment_line()
        self.assertIn("AnimalPoseTracker", line)
        self.assertIn("Python-", line)

    def test_summary_block_aligns_values(self):
        from animalposetracker.training.console import format_summary_block

        block = format_summary_block([("Model", "A"), ("Dataset", "B")])
        lines = block.splitlines()
        self.assertEqual(lines[0], "  Model    A")
        self.assertEqual(lines[1], "  Dataset  B")

    def test_create_prints_banner_with_next_step(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_yaml = _write_minimal_dataset_yaml(root)
            stream = io.StringIO()
            with patch("sys.stdout", stream):
                result = main([
                    "create",
                    "--dataset", str(dataset_yaml),
                    "--dataset-root", str(root / "dataset"),
                    "--workspace", str(root / "workspace"),
                    "--name", "trimouse",
                    "--worker", "test",
                    "--model", "AnimalRTPose",
                    "--scale", "N",
                    "--date", "20261009-120000",
                    "--no-pretrained",
                ])
            self.assertEqual(result, 0)
            banner = stream.getvalue()
            self.assertIn("AnimalPoseTracker", banner)
            self.assertIn("Model", banner)
            self.assertIn("AnimalRTPose-N", banner)
            self.assertIn("Next: animalpose-cli train --config", banner)
            self.assertIn("disabled (train from scratch)", banner)


if __name__ == "__main__":
    unittest.main()
