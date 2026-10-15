import io
import unittest
import tempfile
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from animalposetracker.cli import main
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.training.cli import _resolve_project_default_pretrained
from animalposetracker.training.console import (
    SIMCC_COLUMNS,
    YOLO_POSE_COLUMNS,
    PrettyTrainingRenderer,
    loss_columns,
)
from animalposetracker.training.events import TrainingEvent


class CommandDispatcherTests(unittest.TestCase):
    def test_train_forwards_arguments(self):
        with patch("animalposetracker.training.cli.run", return_value=0) as run:
            self.assertEqual(main(["train", "--config", "project/configs/runtime.yaml"]), 0)
        run.assert_called_once_with(["--config", "project/configs/runtime.yaml"])

    def test_val_selects_validation_only_mode(self):
        with patch("animalposetracker.training.cli.run", return_value=0) as run:
            self.assertEqual(main(["val", "--config", "project/configs/runtime.yaml", "--weights", "best.pt"]), 0)
        run.assert_called_once_with([
            "--validate-only", "--config", "project/configs/runtime.yaml", "--weights", "best.pt",
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
            self.assertEqual(main(["infer", "--poser-weights", "m.onnx", "--source", "0"]), 3)
            inference_cli.run.assert_called_once_with(["--poser-weights", "m.onnx", "--source", "0"])

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
            self.assertTrue((project / "configs" / "runtime.yaml").is_file())
            saved_data = yaml.safe_load((project / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_data["path"], str(dataset_root.resolve()))
            self.assertEqual(saved_data["annotation_format"], "yolo")
            saved_runtime = yaml.safe_load((project / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
            self.assertFalse(
                saved_runtime["train"]["shared"]["runtime"]["pretrained"]
            )

    def test_project_create_keeps_the_templates_flip_table(self):
        # The create flow carries the template's shape keys into the project; flip_idx has to
        # travel with them, or fliplr augmentation mirrors coordinates without swapping the
        # left/right labels.
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_yaml = root / "twoperson.yaml"
            dataset_yaml.write_text(yaml.safe_dump({
                "path": "unused",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [2, 3],
                "flip_idx": [1, 0],
                "names": {0: "person"},
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
                    "--name", "twoperson",
                    "--worker", "test",
                    "--model", "AnimalRTPose",
                    "--scale", "N",
                    "--date", "20261013-190000",
                    "--no-pretrained",
                ])

            project = workspace / "twoperson-test-AnimalRTPose-N-20261013-190000"
            self.assertEqual(result, 0)
            saved_data = yaml.safe_load((project / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_data["flip_idx"], [1, 0])

    def test_no_arguments_prints_help(self):
        with patch("sys.stdout"):
            self.assertEqual(main([]), 0)


class ProjectDatasetShapeTests(unittest.TestCase):
    """A project's keypoint/class inputs belong in dataset.yaml, not only in project.yaml.

    Training reads `kpt_shape` straight out of dataset.yaml, so a project created with the
    built-in default (17 keypoints, 1 class) while project.yaml claimed something else would
    train a model whose output shape never matched its own configuration.
    """

    def _create(self, temporary, **overrides):
        arguments = {
            "local_path": Path(temporary),
            "project_name": "shape",
            "worker": "wux",
            "model_type": "AnimalRTPose",
            "model_scale": "N",
        }
        arguments.update(overrides)
        project = AnimalPoseTrackerProject(**arguments)
        project.create_new_project()
        datasets = project.project_path / "configs" / "dataset.yaml"
        model = project.project_path / "configs" / "model.yaml"
        return (
            yaml.safe_load(datasets.read_text(encoding="utf-8")),
            yaml.safe_load(model.read_text(encoding="utf-8")),
        )

    def test_dataset_and_model_shape_follow_the_project_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, model = self._create(
                temporary,
                keypoints=5,
                visible=True,
                classes=2,
                classes_name=["cat", "dog"],
                skeleton=[[0, 1], [2, 3]],
            )

        self.assertEqual(dataset["kpt_shape"], [5, 3])
        self.assertEqual(dataset["names"], {0: "cat", 1: "dog"})
        self.assertEqual(dataset["skeleton"], [[0, 1], [2, 3]])
        self.assertEqual(dataset["flip_idx"], [0, 1, 2, 3, 4])
        self.assertEqual(len(dataset["kpt_oks_sigmas"]), 5)
        self.assertEqual(model["kpt_shape"], [5, 3])
        self.assertEqual(model["nc"], 2)

    def test_invisible_keypoints_drop_the_visibility_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, model = self._create(
                temporary, keypoints=3, visible=False, classes=1, classes_name=["mouse"]
            )

        self.assertEqual(dataset["kpt_shape"], [3, 2])
        self.assertEqual(model["kpt_shape"], [3, 2])

    def test_class_names_are_padded_to_the_class_count(self):
        # The form holds one name row per class it was told about, but the spin box can ask for
        # more than the rows the user filled in; the mapping still has to cover every class.
        with tempfile.TemporaryDirectory() as temporary:
            dataset, _model = self._create(
                temporary, keypoints=2, visible=True, classes=3, classes_name=["cat"]
            )

        self.assertEqual(dataset["names"], {0: "cat", 1: "class2", 2: "class3"})


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
    saved_runtime = yaml.safe_load((project / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
    return result, saved_runtime


class ProjectCreatePretrainedTests(unittest.TestCase):
    def test_create_with_default_pretrained_does_not_detect_or_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(
                AnimalPoseTrackerProject, "_detect_pretrained",
                side_effect=AssertionError("create must not resolve pretrained weights"),
            ):
                result, saved_runtime = _run_create(root)
            self.assertEqual(result, 0)
            self.assertTrue(saved_runtime["train"]["shared"]["runtime"]["pretrained"])

    def test_create_records_explicit_pretrained_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weights_path = root / "weights" / "animalrtpose-n.pt"
            result, saved_runtime = _run_create(root, "--pretrained", str(weights_path))
            self.assertEqual(result, 0)
            self.assertEqual(
                saved_runtime["train"]["shared"]["runtime"]["pretrained"],
                str(weights_path),
            )

    def test_create_no_pretrained_flag_still_records_false(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result, saved_runtime = _run_create(Path(temporary_directory), "--no-pretrained")
            self.assertEqual(result, 0)
            self.assertFalse(saved_runtime["train"]["shared"]["runtime"]["pretrained"])


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
        header, _, body = output.partition("\n")
        # The column header is printed once for the epoch...
        self.assertIn("Epoch", header)
        self.assertIn("GPU_mem", header)
        self.assertIn("box_loss", header)
        # ...and nothing after it starts a new line: every batch rewinds the same row.
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("\r", body)
        self.assertRegex(body, r"1/2\s+0G\s+22\.5\s+0\.41\s+0\.9")
        self.assertIn("1/9", body)

    def test_known_events_render_tagged_lines_and_a_validation_table(self):
        output = self._render([
            TrainingEvent(
                event="pretrained",
                metrics={"loaded_tensors": 423.0, "shape_mismatches": 42.0},
            ),
            TrainingEvent(event="started", epoch=0, epochs=2, message="device=cuda"),
            TrainingEvent(
                event="validation", epoch=1, epochs=2,
                metrics={"loss": 20.6, "COCO/AP": 0.0, "PCK": 0.0},
            ),
            TrainingEvent(
                event="epoch_end", epoch=1, epochs=2,
                metrics={"train_loss": 22.4, "val_loss": 20.6, "val_COCO/AP": 0.7},
            ),
            TrainingEvent(event="finished", epoch=2, epochs=2),
        ])
        self.assertIn("[pretrained] loaded 423 tensors (42 incompatible shapes skipped)", output)
        self.assertIn("[started] device=cuda", output)
        self.assertIn("[finished]", output)
        self.assertNotIn('"event"', output)
        # An epoch ends with the validation figures and nothing else: the epoch number and
        # the training loss both belong to the progress row above (Ultralytics and MMPose
        # both close an epoch with validation metrics only).
        self.assertNotIn("[epoch", output)
        self.assertNotIn("train_loss", output)
        # A validation event has no line of its own: the epoch's metrics are collected and
        # rendered as one table when the epoch ends, which is what the training loop emits.
        self.assertNotIn("[val]", output)
        self.assertRegex(output, r"Class\s+Images\s+Instances\s+Loss\s+AP\s")
        # The "all" row is where the epoch's validation loss and AP land.
        self.assertRegex(output, r"all\s+-\s+-\s+20\.6\s+0\.7\s")

    def test_unknown_event_falls_back_to_tagged_line(self):
        output = self._render([
            TrainingEvent(event="custom", message="hello", metrics={"x": 1.5}),
        ])
        self.assertIn("[custom] hello x=1.5", output)

    def test_simcc_layout_prints_one_loss_instead_of_five_empty_columns(self):
        stream = io.StringIO()
        renderer = PrettyTrainingRenderer(stream=stream, columns=SIMCC_COLUMNS)
        renderer.emit(TrainingEvent(
            event="batch", epoch=1, epochs=2, step=1, steps=2,
            # Exactly what SimCCKLLoss reports through engine._loss_components.
            metrics={
                "loss": 12.5, "loss_simcc_x": 6.2, "loss_simcc_y": 6.1,
                "loss_simcc_per_sample": 1.5625, "instances": 8, "size": 256,
            },
        ))
        output = stream.getvalue()
        header, _, body = output.partition("\n")
        self.assertIn("loss", header)
        self.assertNotIn("box_loss", header)
        # No column is reserved for a metric the head never reports.
        self.assertNotIn("-", output)
        # The one column is the per-image average, not the batch sum.
        self.assertRegex(body, r"1/2\s+0G\s+1\.562\s+8\s+256")

    def test_simcc_epoch_end_prints_only_the_validation_table(self):
        stream = io.StringIO()
        renderer = PrettyTrainingRenderer(stream=stream, columns=SIMCC_COLUMNS)
        renderer.emit(TrainingEvent(
            event="epoch_end", epoch=1, epochs=2,
            metrics={
                # The loop reports both the KL batch sum and the per-image figure; neither
                # belongs here, and neither does a repeat of the epoch number.
                "train_loss": 12.5,
                "train_loss_simcc_per_sample": 1.5625,
                "val_images": 20, "val_instances": 31,
                "val_COCO/AP": 0.71, "val_PCK": 0.62,
            },
        ))
        output = stream.getvalue()
        self.assertNotIn("[epoch", output)
        self.assertNotIn("train_loss", output)
        self.assertNotIn("12.5", output)
        self.assertNotIn("1.562", output)
        # The SimCC validator scores AP/PCK/AUC/EPE and computes no validation loss.
        self.assertRegex(output, r"Class\s+Images\s+Instances\s+AP\s")
        self.assertIn("PCK", output)
        self.assertNotIn("Loss", output)
        self.assertRegex(output, r"all\s+20\s+31\s+0\.71")
        self.assertNotIn("12.5", output)

    def test_loss_columns_follow_the_head(self):
        self.assertIs(loss_columns("SimCCHead"), SIMCC_COLUMNS)
        self.assertIs(loss_columns("YOLOPoseHead"), YOLO_POSE_COLUMNS)
        self.assertIs(loss_columns(None), YOLO_POSE_COLUMNS)
        self.assertIs(PrettyTrainingRenderer().columns, YOLO_POSE_COLUMNS)


class InferCliTests(unittest.TestCase):
    def test_resolve_num_classes_from_runtime_shape(self):
        from animalposetracker.inference.cli import _resolve_num_classes

        engine = SimpleNamespace(_runtime_output_shapes={"predictions": (1, 43, 8400)})
        self.assertEqual(_resolve_num_classes(engine, (12, 3)), 3)
        empty = SimpleNamespace(_runtime_output_shapes={})
        self.assertEqual(_resolve_num_classes(empty, (12, 3)), 1)

    def test_resolve_num_classes_is_raw_only(self):
        from animalposetracker.inference.cli import _resolve_num_classes
        from animalposetracker.postprocessing.output_decoders import YOLO_POSE_NMS

        # 300 detections over 17x3 keypoints would otherwise be read as 245 classes.
        engine = SimpleNamespace(_runtime_output_shapes={"predictions": (1, 300, 57)})
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            resolved = _resolve_num_classes(engine, (17, 3), default=2, schema=YOLO_POSE_NMS)
        self.assertEqual(resolved, 2)
        self.assertEqual(len(recorded), 1)
        self.assertIn("does not encode the class count", str(recorded[0].message))

    def test_resolve_num_classes_still_derives_for_raw(self):
        from animalposetracker.inference.cli import _resolve_num_classes
        from animalposetracker.postprocessing.output_decoders import YOLO_POSE_RAW

        engine = SimpleNamespace(_runtime_output_shapes={"predictions": (1, 57, 8400)})
        self.assertEqual(_resolve_num_classes(engine, (17, 3), schema=YOLO_POSE_RAW), 2)

    def test_resolve_num_classes_keeps_legacy_behaviour_without_schema(self):
        from animalposetracker.inference.cli import _resolve_num_classes

        engine = SimpleNamespace(_runtime_output_shapes={"predictions": (1, 43, 8400)})
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            self.assertEqual(_resolve_num_classes(engine, (12, 3)), 3)
        self.assertEqual(recorded, [])

    def test_load_dataset_config_returns_the_schema(self):
        from animalposetracker.inference.cli import _load_dataset_config

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_yaml = _write_minimal_dataset_yaml(root)
            names, skeleton, kpt_shape, schema = _load_dataset_config(str(dataset_yaml))
        self.assertEqual(names, ["mouse"])
        self.assertEqual(skeleton, [[0, 1]])
        self.assertEqual(kpt_shape, (2, 3))
        self.assertIsNone(schema)

    def test_load_dataset_config_reads_schema_aliases(self):
        from animalposetracker.inference.cli import _load_dataset_config

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for key in ("schema", "output_schema"):
                with self.subTest(key=key):
                    dataset_yaml = root / f"{key}.yaml"
                    dataset_yaml.write_text(yaml.safe_dump({
                        "kpt_shape": [17, 3],
                        key: "YOLO_Pose_NMS",
                    }), encoding="utf-8")
                    *_, schema = _load_dataset_config(str(dataset_yaml))
                    self.assertEqual(schema, "yolo_pose_nms")

    def test_load_dataset_config_without_a_path(self):
        from animalposetracker.inference.cli import _load_dataset_config

        self.assertEqual(_load_dataset_config(None), ([], [], None, None))

    def test_infer_parser_defaults(self):
        from animalposetracker.inference.cli import _make_argument_parser

        args = _make_argument_parser().parse_args(["--poser-weights", "m.onnx", "--source", "0"])
        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.iou, 0.45)
        self.assertEqual(args.pose_filter, "none")
        self.assertEqual(args.schema, "auto")
        self.assertFalse(args.show)
        self.assertIsNone(args.tracker)

    def test_infer_parser_accepts_yolo_schemas(self):
        from animalposetracker.inference.cli import _make_argument_parser
        from animalposetracker.inference.schema_resolution import SUPPORTED_YOLO_SCHEMAS

        for schema in SUPPORTED_YOLO_SCHEMAS:
            with self.subTest(schema=schema):
                args = _make_argument_parser().parse_args([
                    "--poser-weights", "m.onnx", "--source", "0", "--schema", schema,
                ])
                self.assertEqual(args.schema, schema)

    def test_infer_parser_rejects_reserved_and_unknown_schemas(self):
        from animalposetracker.inference.cli import _make_argument_parser

        for schema in ("mmpose_rtmpose_simcc", "resnet_heatmap"):
            with self.subTest(schema=schema):
                with self.assertRaises(SystemExit):
                    _make_argument_parser().parse_args([
                        "--poser-weights", "m.onnx", "--source", "0", "--schema", schema,
                    ])


class ConsoleBannerTests(unittest.TestCase):
    def test_environment_line_mentions_package_and_python(self):
        from animalposetracker.console import environment_line

        line = environment_line()
        self.assertIn("AnimalPoseTracker", line)
        self.assertIn("Python-", line)

    def test_summary_block_aligns_values(self):
        from animalposetracker.console import format_summary_block

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


class InferenceCliRenderTests(unittest.TestCase):
    """The one-frame renderer the CLI drives -- the ``skeleton``-NameError regression guard.

    ``infer --output`` used to crash before the first frame because the main flow never
    defined the ``skeleton`` argument the renderer takes; the flow now assigns it from
    the session.
    """

    def test_render_accepts_an_empty_skeleton(self):
        import numpy as np

        from animalposetracker.inference.cli import _render

        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        detection = {
            "bbox_xyxy": [4, 4, 20, 20],
            "score": 0.9,
            "class_id": 0,
            "class_name": "person",
            "keypoints": [[6, 6, 1], [10, 10, 1], [14, 14, 1]],
        }

        rendered = _render(frame, [detection], ())

        self.assertEqual(rendered.shape, frame.shape)
        self.assertFalse((rendered == frame).all(), "nothing was drawn")


class InferenceCliTopdownWiringTests(unittest.TestCase):
    """``--detector-weights`` has to switch the stream to the top-down branch.

    Everything behind the switch (the session builder, the ONNX adapter, the
    pipeline) is replaced with recorders: this test owns the wiring, not the
    runtime.
    """

    def _run_infer(self, extra_arguments=()):
        import animalposetracker.inference.cli as cli

        calls = {"session": {}, "detector": {}, "pipeline": {}}

        class _FakeDetector:
            def __init__(self, path, *, conf_threshold, box_format="cxcywh",
                         coord_mode="normalized", preprocess_mode="stretch"):
                calls["detector"] = {
                    "path": Path(path),
                    "conf_threshold": conf_threshold,
                    "box_format": box_format,
                    "coord_mode": coord_mode,
                    "preprocess_mode": preprocess_mode,
                }

        class _FakePipeline:
            def __init__(self, engine, output_config, *, input_mode, input_config,
                         detector, tracker, class_names):
                calls["pipeline"] = {"input_mode": input_mode, "detector": detector}

        def _fake_session(**kwargs):
            calls["session"] = dict(kwargs)
            return SimpleNamespace(
                engine=object(),
                output_config=SimpleNamespace(schema="mmpose_rtmpose_simcc"),
                input_config=object(),
                class_names=("person",),
                skeleton=(),
                head="SimCCHead",
                engine_key="onnx",
            )

        class _FakeCapture:
            def isOpened(self):
                return True

            def read(self):
                return False, None  # an empty source ends the loop after zero frames

            def get(self, _property):
                return 0

            def release(self):
                pass

        argv = ["--poser-weights", "pose.onnx", "--source", "clip.mp4", *extra_arguments]
        with patch.object(cli, "build_inference_session", _fake_session), \
                patch("animalposetracker.inference.detector_adapter.OnnxDetectorAdapter", _FakeDetector), \
                patch("animalposetracker.inference.stream_inference.StreamInferencePipeline", _FakePipeline), \
                patch("cv2.VideoCapture", return_value=_FakeCapture()):
            result = cli.run(argv)
        self.assertEqual(result, 0)
        return calls

    def test_detector_weights_enable_the_topdown_stream(self):
        calls = self._run_infer([
            "--detector-weights", "rtdetr.onnx", "--detector-conf", "0.6",
        ])

        self.assertTrue(calls["session"]["topdown"])
        self.assertEqual(calls["pipeline"]["input_mode"], "topdown")
        self.assertIsNotNone(calls["pipeline"]["detector"])
        self.assertEqual(calls["detector"]["conf_threshold"], 0.6)
        self.assertEqual(calls["detector"]["path"].name, "rtdetr.onnx")

    def test_detector_convention_flags_reach_the_adapter(self):
        calls = self._run_infer([
            "--detector-weights", "yolo26n.onnx",
            "--detector-box-format", "xyxy",
            "--detector-coord-mode", "pixels",
            "--detector-preprocess-mode", "letterbox",
        ])

        detector = calls["detector"]
        self.assertEqual(detector["box_format"], "xyxy")
        self.assertEqual(detector["coord_mode"], "pixels")
        self.assertEqual(detector["preprocess_mode"], "letterbox")

    def test_without_a_detector_the_stream_stays_full_frame(self):
        calls = self._run_infer()

        self.assertFalse(calls["session"]["topdown"])
        self.assertEqual(calls["pipeline"]["input_mode"], "full_frame")
        self.assertIsNone(calls["pipeline"]["detector"])
        self.assertEqual(calls["detector"], {})


if __name__ == "__main__":
    unittest.main()
