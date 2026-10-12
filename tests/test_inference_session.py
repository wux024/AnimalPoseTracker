"""The shared inference assembly: config parsing, artifact inspection, decode contract.

This is the seam the CLI and the GUI's inferencer page both sit on, so these tests pin the
behaviour they are promised: one config parser that accepts either yaml spelling, and one
assembly that decides the schema and the class/keypoint counts the decoder needs.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from animalposetracker.inference.session import (
    build_inference_session,
    instance_to_record,
    load_inference_config,
    read_inference_config,
    validate_inference_selection,
)

PROJECT_STYLE = {
    "project_name": "probe",
    "keypoints": 5,
    "visible": True,
    "keypoints_name": ["a", "b", "c", "d", "e"],
    "classes": 2,
    "classes_name": ["cat", "dog"],
    "skeleton": [[0, 1], [2, 3]],
}

DATASET_STYLE = {
    "names": {0: "cat", 1: "dog"},
    "kpt_shape": [4, 3],
    "skeleton": [[0, 1]],
    "output_schema": "ultralytics_yolo_pose_nms",
}


def _write(directory: str, name: str, mapping: dict) -> Path:
    path = Path(directory) / name
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


class _FakeEngine:
    """Stands in for a real backend so the assembly can run without an artifact."""

    runtime_shapes = {}
    last_kwargs = {}

    def __init__(self, **kwargs):
        _FakeEngine.last_kwargs = kwargs
        self.output_names = tuple(kwargs.get("output_names") or ())
        self._runtime_output_shapes = dict(_FakeEngine.runtime_shapes)
        self.initialised = False

    def model_init(self):
        self.initialised = True


class ConfigParsingTests(unittest.TestCase):
    def test_dataset_style_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(temporary, "dataset.yaml", DATASET_STYLE)
            names, skeleton, kpt_shape, schema = load_inference_config(str(path))

        self.assertEqual(names, ["cat", "dog"])
        self.assertEqual(skeleton, [[0, 1]])
        self.assertEqual(kpt_shape, (4, 3))
        self.assertEqual(schema, "ultralytics_yolo_pose_nms")

    def test_project_style_config_derives_the_keypoint_shape(self):
        # A project yaml has no kpt_shape at all: the shape follows from the keypoint names and
        # the visible flag, which is exactly what the inferencer page hands over.
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(temporary, "project.yaml", PROJECT_STYLE)
            names, skeleton, kpt_shape, schema = load_inference_config(str(path))

        self.assertEqual(names, ["cat", "dog"])
        self.assertEqual(skeleton, [[0, 1], [2, 3]])
        self.assertEqual(kpt_shape, (5, 3))
        self.assertIsNone(schema)

    def test_invisible_keypoints_drop_the_visibility_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(temporary, "project.yaml", {**PROJECT_STYLE, "visible": False})
            _names, _skeleton, kpt_shape, _schema = load_inference_config(str(path))

        self.assertEqual(kpt_shape, (5, 2))

    def test_no_config_stays_lenient_for_the_cli(self):
        # The CLI can be handed a config that only names the classes, with the shape coming from
        # the artifact sidecar, so this reader must not insist on anything.
        self.assertEqual(load_inference_config(None), ([], [], None, None))

    def test_read_inference_config_reports_the_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(temporary, "dataset.yaml", DATASET_STYLE)
            summary = read_inference_config(str(path))

        self.assertEqual(summary.num_classes, 2)
        self.assertEqual(summary.num_keypoints, 4)
        self.assertEqual(summary.schema, "ultralytics_yolo_pose_nms")

    def test_read_inference_config_insists_on_the_class_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(temporary, "broken.yaml", {"kpt_shape": [4, 3]})
            with self.assertRaises(ValueError) as raised:
                read_inference_config(str(path))

        self.assertIn("类别名", str(raised.exception))

    def test_read_inference_config_insists_on_the_keypoint_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write(temporary, "broken.yaml", {"names": {0: "cat"}})
            with self.assertRaises(ValueError) as raised:
                read_inference_config(str(path))

        self.assertIn("关键点数量", str(raised.exception))

    def test_read_inference_config_without_a_path(self):
        with self.assertRaises(ValueError) as raised:
            read_inference_config(None)

        self.assertIn("配置文件", str(raised.exception))


class BuildSessionTests(unittest.TestCase):
    def setUp(self):
        _FakeEngine.runtime_shapes = {}
        _FakeEngine.last_kwargs = {}

    def _build(self, config_mapping=None, metadata=None, **overrides):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "model.onnx"
            weights.write_bytes(b"")
            config_path = (
                _write(temporary, "config.yaml", config_mapping) if config_mapping else None
            )
            arguments = {"weights_path": weights, "config_path": config_path}
            arguments.update(overrides)
            with patch(
                "animalposetracker.artifacts.read_artifact_metadata",
                return_value=dict(metadata or {}),
            ), patch(
                "animalposetracker.artifacts.detect_artifact_format", return_value="onnx"
            ), patch(
                "animalposetracker.inference.session.InferenceEngine", _FakeEngine
            ):
                return build_inference_session(**arguments)

    def test_counts_come_out_of_the_runtime_shape_for_a_raw_artifact(self):
        # 4 box values + 2 class scores + 5 keypoints x 3 = 21 channels.
        _FakeEngine.runtime_shapes = {"output0": (1, 21, 10)}
        session = self._build(
            config_mapping=PROJECT_STYLE,
            metadata={"outputs": ["output0"]},
        )

        self.assertEqual(session.schema, "ultralytics_yolo_pose_raw")
        self.assertEqual(session.output_config.num_classes, 2)
        self.assertEqual(session.output_config.num_keypoints, 5)
        self.assertEqual(session.kpt_shape, (5, 3))
        self.assertEqual(session.class_names, ("cat", "dog"))
        # The backend's own output names have to reach the decoder, or an artifact whose
        # output is not called "predictions" cannot be decoded at all.
        self.assertEqual(session.output_config.output_names, ("output0",))
        self.assertTrue(session.engine.initialised)

    def test_a_declared_schema_wins_over_shape_inference(self):
        _FakeEngine.runtime_shapes = {"output0": (1, 21, 10)}
        session = self._build(
            config_mapping=PROJECT_STYLE,
            metadata={"outputs": ["output0"], "schema": "ultralytics_yolo_pose_nms"},
        )

        self.assertEqual(session.schema, "ultralytics_yolo_pose_nms")
        # Post-NMS rows carry the class id directly, so the configured names are the source.
        self.assertEqual(session.output_config.num_classes, 2)

    def test_the_config_schema_is_used_when_no_argument_is_given(self):
        session = self._build(
            config_mapping=DATASET_STYLE,
            metadata={"outputs": ["output0"]},
        )

        self.assertEqual(session.schema, "ultralytics_yolo_pose_nms")

    def test_an_explicit_schema_argument_wins_over_the_config(self):
        session = self._build(
            config_mapping=DATASET_STYLE,
            metadata={"outputs": ["output0"]},
            schema="ultralytics_yolo_pose_raw",
        )

        self.assertEqual(session.schema, "ultralytics_yolo_pose_raw")

    def test_sidecar_shape_is_used_when_the_config_has_none(self):
        session = self._build(metadata={
            "outputs": ["output0"],
            "kpt_shape": [12, 3],
            "schema": "ultralytics_yolo_pose_raw",
        })

        self.assertEqual(session.kpt_shape, (12, 3))

    def test_an_unknown_keypoint_shape_is_reported(self):
        with self.assertRaises(ValueError) as raised:
            self._build(metadata={"outputs": ["output0"]})

        self.assertIn("kpt_shape is unknown", str(raised.exception))

    def test_a_simcc_artifact_points_at_predict(self):
        with self.assertRaises(ValueError) as raised:
            self._build(metadata={
                "head": "SimCCHead",
                "kpt_shape": [17, 3],
                "outputs": ["simcc_x", "simcc_y"],
            })

        self.assertIn("predict", str(raised.exception))

    def test_the_engine_receives_the_resolved_backend_and_bits(self):
        self._build(
            metadata={"kpt_shape": [4, 3], "schema": "ultralytics_yolo_pose_raw"},
            engine="openvino",
            bits="FP16",
            device="GPU",
        )

        self.assertEqual(_FakeEngine.last_kwargs["engine"], "OpenVINO")
        self.assertEqual(_FakeEngine.last_kwargs["model_bits"], "FP16")
        self.assertEqual(_FakeEngine.last_kwargs["device"], "GPU")


class _FakeInstance:
    def __init__(self, keypoints, scores=None, class_id=1, bbox=(1.0, 2.0, 3.0, 4.0)):
        import numpy as np

        self.keypoints_xy = np.asarray(keypoints, dtype=np.float32)
        self.keypoint_scores = None if scores is None else np.asarray(scores, dtype=np.float32)
        self.keypoints_visible = None
        self.class_id = class_id
        self.score = 0.75
        self.bbox_xyxy = None if bbox is None else np.asarray(bbox, dtype=np.float32)


class RecordShapeTests(unittest.TestCase):
    def test_records_carry_position_and_score_per_keypoint(self):
        record = instance_to_record(
            _FakeInstance([[1.0, 2.0], [3.0, 4.0]], scores=[0.9, 0.1]), ["cat", "dog"]
        )

        self.assertEqual(record["class_id"], 1)
        self.assertEqual(record["class_name"], "dog")
        self.assertEqual(record["score"], 0.75)
        self.assertEqual(record["bbox_xyxy"], [1.0, 2.0, 3.0, 4.0])
        # Coordinates, then the keypoint score as the third column (checked to 3 decimals:
        # the record round-trips through float32).
        self.assertEqual(
            [[round(value, 3) for value in keypoint] for keypoint in record["keypoints"]],
            [[1.0, 2.0, 0.9], [3.0, 4.0, 0.1]],
        )

    def test_a_missing_class_id_falls_back_to_the_first_name(self):
        record = instance_to_record(_FakeInstance([[0.0, 0.0]], class_id=None), ["cat"])

        self.assertEqual(record["class_id"], 0)
        self.assertEqual(record["class_name"], "cat")

    def test_a_missing_box_stays_none(self):
        # The tracker and the renderer both read this field, and the annotator writes None for
        # point-only targets, so the record shape has to preserve it.
        record = instance_to_record(_FakeInstance([[0.0, 0.0]], bbox=None), ["cat"])

        self.assertIsNone(record["bbox_xyxy"])

    def test_records_are_json_serialisable(self):
        record = instance_to_record(_FakeInstance([[1.5, 2.5]], scores=[0.5]), ["cat", "dog"])

        self.assertEqual(json.loads(json.dumps(record))["class_name"], "dog")


class ValidationTests(unittest.TestCase):
    """What the page checks the moment a file is picked, before anything is loaded."""

    def _pair(self, config_mapping, metadata=None, sidecar_text=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config_path = _write(directory.name, "config.yaml", config_mapping)
        weights_path = Path(directory.name) / "model.onnx"
        weights_path.write_bytes(b"")
        if metadata is not None or sidecar_text is not None:
            payload = sidecar_text if sidecar_text is not None else json.dumps(metadata)
            Path(directory.name, "model.onnx.json").write_text(payload, encoding="utf-8")
        return str(config_path), str(weights_path)

    def test_a_matching_pair_passes(self):
        config, weights = self._pair(DATASET_STYLE, {"kpt_shape": [4, 3], "head": "YOLOPoseHead"})
        report = validate_inference_selection(config, weights)

        self.assertEqual(report.errors, ())
        self.assertEqual(report.warnings, ())
        self.assertTrue(report.config_ok)
        self.assertTrue(report.weights_ok)

    def test_a_config_without_class_names_is_refused_in_chinese(self):
        config, _weights = self._pair({"kpt_shape": [4, 3]})
        report = validate_inference_selection(config, None)

        self.assertIn("类别名", " ".join(report.errors))
        self.assertFalse(report.config_ok)
        self.assertFalse(report.weights_ok)  # nothing to run yet

    def test_a_config_without_a_keypoint_count_is_refused(self):
        config, _weights = self._pair({"names": {0: "cat"}})
        report = validate_inference_selection(config, None)

        self.assertIn("关键点数量", " ".join(report.errors))

    def test_a_simcc_artifact_is_refused_while_selecting(self):
        config, weights = self._pair(DATASET_STYLE, {"head": "SimCCHead", "kpt_shape": [4, 3]})
        report = validate_inference_selection(config, weights)

        self.assertIn("predict", " ".join(report.errors))
        self.assertTrue(report.config_ok)
        self.assertFalse(report.weights_ok)

    def test_a_simcc_artifact_is_caught_from_its_output_names_too(self):
        config, weights = self._pair(DATASET_STYLE, {"outputs": ["simcc_x", "simcc_y"]})
        report = validate_inference_selection(config, weights)

        self.assertIn("predict", " ".join(report.errors))

    def test_an_artifact_without_a_sidecar_only_warns(self):
        config, weights = self._pair(DATASET_STYLE)
        report = validate_inference_selection(config, weights)

        self.assertEqual(report.errors, ())
        self.assertTrue(report.weights_ok)
        self.assertIn("sidecar", " ".join(report.warnings))

    def test_a_keypoint_count_mismatch_only_warns(self):
        # The decoder follows the config, so this shifts the keypoint block rather than failing;
        # the user needs to know, but it does not block the run.
        config, weights = self._pair(DATASET_STYLE, {"kpt_shape": [12, 3]})
        report = validate_inference_selection(config, weights)

        self.assertEqual(report.errors, ())
        self.assertIn("不一致", " ".join(report.warnings))

    def test_a_skeleton_index_beyond_the_keypoints_warns(self):
        config, _weights = self._pair({**DATASET_STYLE, "skeleton": [[0, 1], [3, 9]]})
        report = validate_inference_selection(config, None)

        self.assertEqual(report.errors, ())
        self.assertIn("越界", " ".join(report.warnings))
        self.assertIn("9", " ".join(report.warnings))

    def test_a_corrupt_sidecar_becomes_a_message(self):
        config, weights = self._pair(DATASET_STYLE, sidecar_text="{not json at all")
        report = validate_inference_selection(config, weights)

        self.assertIn("无法解析", " ".join(report.errors))
        self.assertFalse(report.weights_ok)


if __name__ == "__main__":
    unittest.main()
