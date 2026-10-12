"""Headless checks for the PySide6 pages and the Qt threads they drive.

Offscreen Qt keeps these runnable without a display, and the module is skipped outright when
PySide6 is missing -- it ships as the optional ``gui`` extra, so the core suite must stay
green without it.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml

from animalposetracker.inference.session import instance_to_record, pixel_points
from animalposetracker.postprocessing.output_decoders import (
    YOLO_POSE_RAW,
    PoseOutputConfig,
    decode_pose_outputs,
)
from animalposetracker.postprocessing.workflows import filter_tracked_video_records

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QStyleOptionViewItem

    from animalposetracker.gui.annotatorpage import AnnotatorPage
    from animalposetracker.gui.createnewprojectpage import CreateNewProjectPage
    from animalposetracker.gui.inferencerpage import InferencerPage
    from animalposetracker.gui.mainpage import AnimalPoseTrackerPage
    from animalposetracker.utils import VideoWriterThread
except ImportError:  # pragma: no cover - depends on the optional gui extra
    QApplication = None
    AnnotatorPage = None
    CreateNewProjectPage = None
    InferencerPage = None
    AnimalPoseTrackerPage = None
    VideoWriterThread = None


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class NewProjectFormTests(unittest.TestCase):
    """What the new-project form collects has to reach the configs the trainer reads."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_form_inputs_reach_dataset_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = CreateNewProjectPage()
            page.ProjectConfig.setText("formprobe")
            page.WorkerConfig.setText("wux")
            page.LocationPathDisplay.setText(temporary)
            page.KeypointConfig.setValue(6)
            page.ClassConfig.setValue(2)
            page.VisibleSelection.setChecked(True)

            page.onCreateProjectClicked()

            dataset_path = page.project.project_path / "configs" / "dataset.yaml"
            dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
            model = yaml.safe_load(
                (page.project.project_path / "configs" / "model.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(dataset["kpt_shape"], [6, 3])
        self.assertEqual(len(dataset["names"]), 2)
        self.assertEqual(model["kpt_shape"], [6, 3])
        self.assertEqual(model["nc"], 2)


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class InferencerControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_start_enables_once_a_config_and_weights_are_both_chosen(self):
        # The old guard tested an attribute that is never assigned, so Start stayed disabled
        # after a weights file was picked -- the button that runs inference never lit up.
        # The config has to be one that passes the selection check: class names and a keypoint
        # count. The weights carry no sidecar, which is only a warning.
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "dataset.yaml"
            config_path.write_text("names: {0: probe}\nkpt_shape: [2, 3]\n", encoding="utf-8")
            weights_path = Path(temporary) / "best.onnx"
            weights_path.write_bytes(b"")

            page = InferencerPage()
            self.assertFalse(page.Start.isEnabled())

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ):
                page.onSelectConfigureClicked()
            self.assertTrue(page._config_ready)
            self.assertFalse(page.Start.isEnabled())

            def fake_engine_detection(_extension):
                # Stand in for the real hardware probe: one engine with one device.
                page.supported_engines_and_devices = {"ONNX": ["Intel CPU"]}

            with patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(weights_path), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(
                page, "_init_engine_selection"
            ), patch.object(
                page, "_init_available_devices"
            ), patch.object(
                page, "_init_model_bits"
            ):
                page.onSelectWeightsClicked()

            self.assertTrue(page._weights_ready)
            self.assertTrue(page.Start.isEnabled())

    def test_a_broken_config_keeps_start_dark_and_says_why(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "dataset.yaml"
            config_path.write_text("kpt_shape: [2, 3]\n", encoding="utf-8")

            page = InferencerPage()
            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ), patch("animalposetracker.gui.inferencerpage.QMessageBox") as box:
                page.onSelectConfigureClicked()

            self.assertFalse(page._config_ready)
            self.assertFalse(page.Start.isEnabled())
            self.assertIn("类别名", box.critical.call_args.args[2])

    def test_reselecting_a_broken_config_turns_start_off_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            good = Path(temporary) / "dataset.yaml"
            good.write_text("names: {0: probe}\nkpt_shape: [2, 3]\n", encoding="utf-8")
            bad = Path(temporary) / "project.yaml"
            bad.write_text("project_name: probe\n", encoding="utf-8")
            weights_path = Path(temporary) / "best.onnx"
            weights_path.write_bytes(b"")

            page = InferencerPage()
            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(good), ""),
            ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
                page.onSelectConfigureClicked()
            self.assertTrue(page._config_ready)

            def fake_engine_detection(_extension):
                page.supported_engines_and_devices = {"ONNX": ["Intel CPU"]}

            with patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(weights_path), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"):
                page.onSelectWeightsClicked()
            self.assertTrue(page.Start.isEnabled())

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(bad), ""),
            ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
                page.onSelectConfigureClicked()

            self.assertFalse(page._config_ready)
            self.assertFalse(page.Start.isEnabled())

    def test_a_simcc_artifact_is_refused_before_the_hardware_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "dataset.yaml"
            config_path.write_text("names: {0: probe}\nkpt_shape: [2, 3]\n", encoding="utf-8")
            weights_path = Path(temporary) / "best.onnx"
            weights_path.write_bytes(b"")
            Path(temporary, "best.onnx.json").write_text(
                '{"head": "SimCCHead", "kpt_shape": [2, 3]}', encoding="utf-8"
            )

            page = InferencerPage()
            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
                page.onSelectConfigureClicked()

            with patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(weights_path), ""),
            ), patch.object(page, "_get_supported_engines_and_devices") as probe, patch(
                "animalposetracker.gui.inferencerpage.QMessageBox"
            ) as box:
                page.onSelectWeightsClicked()

            self.assertFalse(page._weights_ready)
            self.assertFalse(page.Start.isEnabled())
            self.assertIn("predict", box.critical.call_args.args[2])
            # The check runs first, so the engine/device probe never happens.
            probe.assert_not_called()


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class ConfigureTreeTests(unittest.TestCase):
    """The parameter trees must only offer keys the backend actually reads.

    A row whose key is not in the project's other.yaml raises the moment it is edited, and a
    row whose key nothing consumes is a knob that silently does nothing. The trees are generated
    from the Designer files, which mirror the full set of reference-toolkit options, so mainpage
    prunes them at startup.
    """

    # The exact rows each tree should end up with. Editing this list is the point: it forces
    # every addition or removal to be a deliberate decision.
    EXPECTED = {
        "TrainingConfigureEdit": {
            "model", "data", "epochs", "time", "patience", "batch", "imgsz", "save",
            "save_period", "cache", "device", "workers", "project", "name", "pretrained",
            "optimizer", "seed", "cos_lr", "close_mosaic", "resume", "amp", "fraction",
            "profile", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl", "pose", "kobj", "nbs",
            "val", "plots",
        },
        "EvaluateConfigure": {
            "data", "batch", "imgsz", "conf", "iou", "max_det", "half", "device", "dnn",
            "plots", "rect", "split", "project", "name", "profile", "verbose", "workers",
            "agnostic_nms", "single_cls",
        },
        "InferenceConfigure": {
            "source", "prediction_confidence", "prediction_iou", "imgsz", "half", "device",
            "batch", "max_det", "agnostic_nms", "classes", "project", "name", "verbose",
            "save", "show_conf", "show_boxes", "line_width", "kpt_radius", "kpt_line",
            # Added by mainpage: the prediction pipeline reads these, the Designer file has no
            # row for them, and the dotted names address the nested mappings in other.yaml.
            "tracking.enabled", "tracking.algorithm",
            "pose_filter.method", "pose_filter.window_length", "pose_filter.polyorder",
            "pose_filter.max_gap",
        },
        # Untouched: every export option is read by the export pipeline.
        "ExportConfigure": {
            "format", "imgsz", "keras", "optimize", "half", "int8", "dynamic", "simplify",
            "opset", "workspace", "nms", "batch", "device", "data", "fraction",
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def _keys(self, tree):
        return {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}

    def test_each_tree_offers_exactly_the_supported_keys(self):
        page = AnimalPoseTrackerPage()
        for tree_name, expected in self.EXPECTED.items():
            with self.subTest(tree=tree_name):
                self.assertEqual(self._keys(getattr(page, tree_name)), expected)

    def test_prediction_thresholds_are_fully_qualified(self):
        # `conf`/`iou` are read as validation thresholds, not prediction ones; prediction reads
        # prediction_confidence/prediction_iou, so those are the keys the tree has to carry.
        page = AnimalPoseTrackerPage()
        keys = self._keys(page.InferenceConfigure)
        self.assertNotIn("conf", keys)
        self.assertNotIn("iou", keys)

    def test_editing_any_offered_row_does_not_raise(self):
        page = AnimalPoseTrackerPage()
        for tree_name in self.EXPECTED:
            tree = getattr(page, tree_name)
            for index in range(tree.topLevelItemCount()):
                item = tree.topLevelItem(index)
                with self.subTest(tree=tree_name, key=item.text(0)):
                    item.setText(1, "1")

    @staticmethod
    def _row(tree, key):
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if item.text(0) == key:
                return item
        raise AssertionError(f"{key} is not in the tree")

    def test_editing_a_nested_row_updates_the_nested_mapping(self):
        # `other.yaml` holds tracking and pose_filter as mappings, so the rows address one field
        # of each; writing the row has to land inside the mapping the pipeline reads.
        page = AnimalPoseTrackerPage()
        page.project.other_config["tracking"] = {"enabled": False, "algorithm": "bytetrack"}
        page.project.other_config["pose_filter"] = {"method": "none", "window_length": 5}
        page.initialize_other_config()

        tree = page.InferenceConfigure
        self.assertEqual(self._row(tree, "tracking.enabled").text(1), "False")
        self._row(tree, "tracking.enabled").setText(1, "True")
        self._row(tree, "pose_filter.window_length").setText(1, "9")

        self.assertIs(page.project.other_config["tracking"]["enabled"], True)
        self.assertEqual(page.project.other_config["pose_filter"]["window_length"], 9)

    def test_a_project_predating_the_nested_keys_gets_editable_rows(self):
        # Projects saved before these keys existed must still be able to turn tracking on.
        page = AnimalPoseTrackerPage()
        page.project.other_config.clear()
        page.initialize_other_config()

        self.assertEqual(page.project.other_config["tracking"]["enabled"], False)
        self.assertEqual(page.project.other_config["pose_filter"]["method"], "none")

        self._row(page.InferenceConfigure, "tracking.enabled").setText(1, "True")
        self.assertIs(page.project.other_config["tracking"]["enabled"], True)

    @staticmethod
    def _editor(page, tree, key):
        for index in range(tree.topLevelItemCount()):
            if tree.topLevelItem(index).text(0) == key:
                model_index = tree.model().index(index, 1)
                return page._configure_delegate.createEditor(
                    page, QStyleOptionViewItem(), model_index
                )
        raise AssertionError(f"{key} is not in the tree")

    def test_fixed_set_rows_offer_their_values(self):
        page = AnimalPoseTrackerPage()
        self.assertEqual(
            page.configure_choices("tracking.algorithm"),
            ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack"),
        )
        self.assertEqual(page.configure_choices("pose_filter.method"), ("none", "median", "savgol"))
        self.assertEqual(page.configure_choices("split"), ("train", "val", "test"))
        # `cache` accepts two words besides a bool, so it is listed rather than derived.
        self.assertEqual(page.configure_choices("cache"), ("True", "False", "ram", "disk"))
        # Numbers and paths stay free text.
        self.assertEqual(page.configure_choices("epochs"), ())
        self.assertEqual(page.configure_choices("source"), ())

    def test_boolean_rows_are_recognised_from_the_config_value(self):
        page = AnimalPoseTrackerPage()
        page.project.other_config["tracking"] = {"enabled": False}
        page.project.other_config["epochs"] = 10

        self.assertEqual(page.configure_choices("tracking.enabled"), ("True", "False"))
        self.assertEqual(page.configure_choices("epochs"), ())

    def test_rows_that_also_take_text_keep_the_text_editor(self):
        # `pretrained` names a checkpoint when it holds a string, so a True/False dropdown would
        # take away a capability rather than add one.
        page = AnimalPoseTrackerPage()
        self.assertEqual(page.configure_choices("pretrained"), ())

    def test_the_value_column_edits_with_a_dropdown_when_there_are_choices(self):
        page = AnimalPoseTrackerPage()
        page.project.other_config["tracking"] = {"enabled": True, "algorithm": "bytetrack"}
        page.initialize_other_config()
        tree = page.InferenceConfigure

        selected = self._editor(page, tree, "tracking.enabled")
        self.assertIsInstance(selected, QComboBox)
        self.assertEqual(
            [selected.itemText(index) for index in range(selected.count())], ["True", "False"]
        )
        # A row with no fixed set still opens the plain text editor.
        self.assertIsInstance(self._editor(page, tree, "imgsz"), QLineEdit)


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class InferencerDecodeTests(unittest.TestCase):
    """The page decodes and draws through the shared layers.

    It used to call the engine's own `postprocess`, a raw-layout hard slicer whose only remaining
    caller was this page -- so its results were right for one of the three ultralytics layouts and
    silently wrong for the other two. These tests pin the shared contract instead.
    """

    CONFIG = PoseOutputConfig(
        schema=YOLO_POSE_RAW,
        num_classes=1,
        num_keypoints=2,
        keypoint_dims=3,
        output_names=("predictions",),
    )

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _tensor():
        """One anchor: a box, one class score, two keypoints (the second one unscorable)."""
        array = np.zeros((1, 4 + 1 + 2 * 3, 1), dtype=np.float32)
        array[0, 0:4, 0] = [0.5, 0.5, 0.2, 0.2]
        array[0, 4, 0] = 0.9
        array[0, 5:11, 0] = [0.1, 0.2, 0.9, 0.3, 0.4, 0.0]
        return {"predictions": array}

    def _page(self, **config):
        page = InferencerPage()
        page.session = SimpleNamespace(
            engine=SimpleNamespace(
                output_names=("predictions",),
                preprocess=lambda frame: (frame, None),
            ),
            output_config=self.CONFIG,
            input_config=SimpleNamespace(input_size=(640, 480)),
            class_names=("mouse",),
            skeleton=((0, 1),),
            kpt_shape=(2, 3),
        )
        page._init_visualization_config()
        page.visualization_config.update(config)
        page._build_palettes()
        return page

    def test_decode_matches_the_shared_decoder_point_for_point(self):
        page = self._page()
        decoded = page._decode(self._tensor(), None)["instances"]
        direct = decode_pose_outputs(self._tensor(), self.CONFIG, inverse_affine=None)

        self.assertEqual(len(decoded), len(direct))
        self.assertTrue(np.allclose(decoded[0].keypoints_xy, direct[0].keypoints_xy))
        self.assertTrue(np.allclose(decoded[0].bbox_xyxy, direct[0].bbox_xyxy))
        self.assertEqual(decoded[0].score, direct[0].score)
        self.assertEqual(decoded[0].class_id, direct[0].class_id)

    def test_the_pipeline_runs_the_shared_functions(self):
        page = self._page()
        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.cv2.VideoCapture", return_value=_FakeCapture()
        ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
            page._init_inference_threads()

        # Bound methods compare equal by (instance, function); they are not identical objects.
        self.assertEqual(page.preprocess_thread.preprocess_function, page.session.engine.preprocess)
        self.assertEqual(page.inference_thread.inference_function, page._infer_outputs)
        self.assertEqual(page.postprocess_thread.postprocess_function, page._decode)
        self.assertEqual(page.visualize_thread.visualize_function, page._present)

    def test_named_outputs_are_what_the_decoder_is_handed(self):
        page = self._page()
        seen = {}

        def named_outputs(image, output_names=None):
            seen["output_names"] = output_names
            return {"predictions": np.zeros((1, 10, 1), dtype=np.float32)}

        page.session.engine.inference_named_outputs = named_outputs
        page._infer_outputs("image")

        # Without the artifact's own output names, an artifact whose output is not called
        # "predictions" cannot be decoded at all.
        self.assertEqual(seen["output_names"], ("predictions",))

    def test_records_use_the_shared_record_shape(self):
        page = self._page()
        results = page._decode(self._tensor(), None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _rendered, presented = page._present(frame, dict(results))

        expected = instance_to_record(results["instances"][0], ["mouse"])
        self.assertEqual(presented["records"], [expected])

    def test_an_unscorable_keypoint_is_not_drawn(self):
        page = self._page()
        results = page._decode(self._tensor(), None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _rendered, presented = page._present(frame, dict(results))

        # The second keypoint scores zero, so it keeps no position: not drawn, and no edge
        # reaches it. The record still carries its row, with the score in the third column.
        self.assertEqual(
            pixel_points(presented["records"][0]["keypoints"]), [(0, 0), None]
        )

    def test_the_background_selector_replaces_the_frame(self):
        # With every overlay switched off, the frame is exactly what the selector says.
        quiet = dict(
            show_fps=False,
            show_preprocess_time=False,
            show_inference_time=False,
            show_postprocess_time=False,
        )
        white = self._page(background="White", **quiet)
        painted = white._present(np.zeros((24, 32, 3), np.uint8), {"instances": []})[0]
        self.assertTrue(bool(np.all(painted == 255)))

        black = self._page(background="Black", **quiet)
        painted = black._present(np.full((24, 32, 3), 255, np.uint8), {"instances": []})[0]
        self.assertTrue(bool(np.all(painted == 0)))

    def test_the_timing_overlay_follows_its_switches(self):
        results = {
            "instances": [],
            "preprocess_time": 0.001,
            "inference_time": 0.002,
            "postprocess_time": 0.003,
            "fps": 166.0,
        }

        def overlay_lines(page):
            lines = []
            with patch(
                "animalposetracker.gui.inferencerpage.cv2.putText",
                lambda _frame, text, *_rest: lines.append(text),
            ):
                page._present(np.zeros((480, 640, 3), np.uint8), dict(results))
            return lines

        every = self._page(show_fps=True, show_preprocess_time=True,
                           show_inference_time=True, show_postprocess_time=True)
        # Single prefixes: the engine's renderer printed them twice ("FPS: FPS: 30.0 FPS").
        self.assertEqual(overlay_lines(every), [
            "FPS: 166.0",
            "Preprocess: 1.000 ms",
            "Inference: 2.000 ms",
            "Postprocess: 3.000 ms",
        ])

        only_fps = self._page(show_fps=True, show_preprocess_time=False,
                              show_inference_time=False, show_postprocess_time=False)
        self.assertEqual(overlay_lines(only_fps), ["FPS: 166.0"])

        silent = self._page(show_fps=False, show_preprocess_time=False,
                            show_inference_time=False, show_postprocess_time=False)
        self.assertEqual(overlay_lines(silent), [])

    def test_palettes_follow_the_session(self):
        page = self._page()

        self.assertEqual(len(page._class_colors), 1)
        self.assertEqual(len(page._edge_colors), 1)
        self.assertEqual(len(page._keypoint_colors), 2)

    def test_records_are_collected_only_when_the_switch_is_on(self):
        page = self._page()
        page.SaveRecords.setChecked(True)
        for _ in range(3):
            page._present(np.zeros((480, 640, 3), np.uint8), page._decode(self._tensor(), None))

        self.assertEqual([entry["frame"] for entry in page.records], [0, 1, 2])
        self.assertEqual(page._frame_index, 3)
        # The shape the `infer --records` file carries, so a workbench can read both.
        self.assertEqual(sorted(page.records[0]), ["frame", "predictions"])
        self.assertEqual(
            sorted(page.records[0]["predictions"][0]),
            ["bbox_xyxy", "class_id", "class_name", "keypoints", "score"],
        )

        quiet = self._page()
        quiet._present(np.zeros((480, 640, 3), np.uint8), quiet._decode(self._tensor(), None))
        self.assertEqual(quiet.records, [])

    def test_the_record_switch_joins_the_start_stop_allow_lists(self):
        page = self._page()

        page.tools_disabled()
        self.assertFalse(page.SaveRecords.isEnabled())
        page.tools_enabled()
        self.assertTrue(page.SaveRecords.isEnabled())
        # Built at runtime, so it has to be parented into the group by hand.
        self.assertEqual(page.SaveRecords.parentWidget().objectName(), "ShowToolsGroup")

    def test_the_records_file_appears_only_when_the_switch_asked_for_it(self):
        page = self._page()
        page.records = [{"frame": 0, "predictions": []}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            Path, "cwd", return_value=Path(temporary)
        ):
            target = Path(temporary) / "runs" / "inference" / "records.json"

            page.SaveRecords.setChecked(False)
            page._finalize_cleanup()
            self.assertFalse(target.exists())

            page.SaveRecords.setChecked(True)
            page._finalize_cleanup()
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), page.records)

    def test_tracking_controls_offer_the_cli_algorithms(self):
        from animalposetracker.tracking.config import ALGORITHMS

        page = self._page()
        self.assertEqual(
            [page.TrackerAlgorithm.itemText(index) for index in range(page.TrackerAlgorithm.count())],
            list(ALGORITHMS),
        )

    def test_the_tracking_controls_follow_the_start_stop_allow_lists(self):
        page = self._page()

        page.tools_disabled()
        self.assertFalse(page.EnableTracking.isEnabled())
        self.assertFalse(page.TrackerAlgorithm.isEnabled())

        page.tools_enabled()
        self.assertTrue(page.EnableTracking.isEnabled())
        # The picker only means anything while tracking is on.
        self.assertFalse(page.TrackerAlgorithm.isEnabled())
        page.EnableTracking.setChecked(True)
        self.assertTrue(page.TrackerAlgorithm.isEnabled())

    def test_no_tracker_is_built_unless_the_switch_is_set(self):
        page = self._page()
        page._build_tracker()
        self.assertIsNone(page.tracker)

        page.tools_enabled()
        page.EnableTracking.setChecked(True)
        page.TrackerAlgorithm.setCurrentText("bytetrack")
        page._build_tracker()
        self.assertEqual(type(page.tracker).__name__, "ByteTrack")

    def test_track_ids_stay_with_one_target_across_frames(self):
        page = self._page()
        page.tools_enabled()
        page.EnableTracking.setChecked(True)
        page._build_tracker()
        page.SaveRecords.setChecked(True)

        identifiers = []
        for _ in range(3):
            _rendered, results = page._present(
                np.zeros((480, 640, 3), np.uint8), page._decode(self._tensor(), None)
            )
            identifiers.append(results["records"][0]["track_id"])

        self.assertIsNotNone(identifiers[0])
        self.assertEqual(len(set(identifiers)), 1)
        self.assertEqual([entry["frame"] for entry in page.records], [0, 1, 2])
        # Tracking adds a key to the record; the keypoints keep their score column, which is
        # what the renderer decides visibility from.
        self.assertEqual(len(results["records"][0]["keypoints"][0]), 3)

    def test_the_label_carries_the_track_id(self):
        page = self._page()
        page.tools_enabled()
        page.EnableTracking.setChecked(True)
        page._build_tracker()
        page.visualization_config.update({"show_classes": True})

        labels = []
        with patch(
            "animalposetracker.gui.inferencerpage.draw_instance",
            side_effect=lambda *arguments, **keywords: labels.append(arguments[2]),
        ):
            for _ in range(2):
                page._present(np.zeros((480, 640, 3), np.uint8), page._decode(self._tensor(), None))

        self.assertEqual(len(labels), 2)
        self.assertTrue(labels[0].startswith("#"))
        self.assertIn("mouse: 0.90", labels[0])
        self.assertEqual(labels[0].split()[0], labels[1].split()[0])

    def _tracked_page(self, **config):
        page = self._page(**config)
        page.tools_enabled()
        page.EnableTracking.setChecked(True)
        page._build_tracker()
        page.SaveRecords.setChecked(True)
        return page

    def _feed(self, page, frames=6):
        """Feed jittered frames of one target, which is what tracking has to follow."""
        for index in range(frames):
            tensor = self._tensor()
            tensor["predictions"][0, 0, 0] += 0.01 * (index % 2)
            page._present(np.zeros((480, 640, 3), np.uint8), page._decode(tensor, None))

    def test_the_filter_controls_match_the_command_line_defaults(self):
        page = self._page()

        # The switch is the on/off, so the method list has no `none` of its own.
        self.assertFalse(page.EnableFilter.isChecked())
        self.assertEqual(
            [page.FilterMethod.itemText(index) for index in range(page.FilterMethod.count())],
            ["median", "savgol"],
        )
        self.assertEqual(page.FilterWindow.value(), 5)
        self.assertEqual(page.FilterPolyorder.value(), 2)
        self.assertEqual(page.FilterConfidence.value(), 0.25)
        self.assertEqual(page.FilterMaxGap.value(), 0)

    def test_the_filter_controls_follow_tracking_and_their_own_switch(self):
        page = self._page()
        page.tools_disabled()
        self.assertFalse(page.EnableFilter.isEnabled())

        page.tools_enabled()
        # Filtering needs track ids, which only exist while tracking is on.
        self.assertFalse(page.EnableFilter.isEnabled())
        page.EnableTracking.setChecked(True)
        self.assertTrue(page.EnableFilter.isEnabled())
        # Its own switch decides whether its values matter.
        self.assertFalse(page.FilterMethod.isEnabled())
        page.EnableFilter.setChecked(True)
        self.assertTrue(page.FilterMethod.isEnabled())
        self.assertTrue(page.FilterWindow.isEnabled())
        # The polynomial order belongs to Savitzky-Golay.
        self.assertFalse(page.FilterPolyorder.isEnabled())
        page.FilterMethod.setCurrentText("savgol")
        self.assertTrue(page.FilterPolyorder.isEnabled())

    def test_the_filter_produces_what_the_workflow_produces(self):
        page = self._tracked_page()
        page.EnableFilter.setChecked(True)
        page.FilterMethod.setCurrentText("savgol")
        page.FilterWindow.setValue(3)
        page.FilterConfidence.setValue(0.5)
        self._feed(page)

        expected = filter_tracked_video_records(
            page.records,
            method="savgol",
            window_length=3,
            polyorder=2,
            confidence_threshold=0.5,
            max_gap=0,
        )
        self.assertEqual(page._final_records(), expected)

    def test_without_the_switch_the_records_are_written_untouched(self):
        page = self._tracked_page()
        self._feed(page)

        self.assertIs(page._final_records(), page.records)  # not even copied

    def test_turning_the_filter_on_alone_is_not_enough_without_tracking(self):
        page = self._page()
        page.tools_enabled()
        page.SaveRecords.setChecked(True)
        page.FilterMethod.setCurrentText("median")
        self._feed(page)

        # The switch is checked, but there are no track ids for the filter to work on.
        self.assertFalse(page._filter_is_on())
        self.assertIs(page._final_records(), page.records)

    def test_the_layout_follows_the_third_plan(self):
        page = self._page()
        left = page.BaseToolsLayout

        # Row 2 holds the decode thresholds, moved out of the display group with their widgets.
        self.assertIs(left.itemAt(1).layout(), page.ThresholdLayout)
        self.assertEqual(page.IoU.parentWidget().objectName(), "BaseToolsGroup")
        self.assertEqual(page.Conf.parentWidget().objectName(), "BaseToolsGroup")
        # Rows 3 and 4 hold the two optional stages and the filter's values.
        self.assertIs(left.itemAt(2).layout(), page.TrackerFilterLayout)
        self.assertIs(left.itemAt(3).layout(), page.FilterValuesLayout)
        self.assertEqual(
            [left.itemAt(2).layout().itemAt(index).widget() for index in range(4)],
            [page.EnableTracking, page.TrackerAlgorithm, page.EnableFilter, page.FilterMethod],
        )
        # Below them the group keeps its own order: source, camera parameters, engine and device.
        self.assertIs(left.itemAt(4).layout(), page.CameraVideosLayout)
        self.assertIs(left.itemAt(5).layout(), page.CameraParametersLayout)
        self.assertIs(left.itemAt(6).layout(), page.EngineDeviceLayout)
        self.assertIs(left.itemAt(7).layout(), page.StartEndLayout)
        self.assertEqual(page.EngineDeviceLayout.count(), 3)

        # The display group opens with the two switches that write files.
        right = page.ShowToolsGroupVLayout
        self.assertIs(right.itemAt(0).layout(), page.SaveSwitchLayout)
        self.assertEqual(
            [right.itemAt(0).layout().itemAt(index).widget() for index in range(2)],
            [page.Save, page.SaveRecords],
        )
        # The row those two used to share is gone.
        self.assertEqual(right.count(), 7)

    def test_every_row_shares_one_cell_width_and_packs_left(self):
        # Qt spreads a row's leftover space, so rows of different lengths never line up and a lone
        # spin box stretches to a combo's width. One cell width plus a trailing stretch fixes both.
        page = self._page()
        width = InferencerPage._CELL_WIDTH

        for attribute in (
            "SelectConfigure", "SelectWeights", "WidthSetup",
            "ModelBits", "EngineSelection", "DeviceSelection", "Start", "End",
            "IoU", "Conf", "TrackerAlgorithm", "FilterMethod", "FilterWindow",
            "Save", "SaveRecords", "Show", "FPSShow", "ShowBBox", "EnableTracking",
        ):
            with self.subTest(widget=attribute):
                self.assertEqual(getattr(page, attribute).maximumWidth(), width)

        # The source field shows a path, so it is deliberately wider than one cell.
        self.assertEqual(page.CameraVideosSelection.maximumWidth(), 240)

        for layout in (
            page.BaseConfigureLayout, page.ThresholdLayout, page.TrackerFilterLayout,
            page.FilterValuesLayout, page.CameraVideosLayout, page.CameraParametersLayout,
            page.StartEndLayout, page.SaveSwitchLayout, page.MainShowLayout, page.ShowInfoLayput,
        ):
            with self.subTest(layout=layout.objectName()):
                self.assertIsNotNone(layout.itemAt(layout.count() - 1).spacerItem())

        # The slider rows keep their sliders full width, so they must not be packed.
        for layout in (
            page.FontScaleLayout, page.ShowKeypointsLayout, page.ShowSkeletonsLayout,
            page.ShowBBoxLayout,
        ):
            with self.subTest(layout=layout.objectName()):
                self.assertIsNone(layout.itemAt(layout.count() - 1).spacerItem())

    def test_the_two_save_switches_say_what_they_save(self):
        # "Save" next to "Save Records" left the reader guessing which saved what.
        page = self._page()

        self.assertEqual(page.Save.text(), "Save Video")
        self.assertIn("output.avi", page.Save.toolTip())
        self.assertEqual(page.SaveRecords.text(), "Save Records")
        self.assertIn("records.json", page.SaveRecords.toolTip())

    def test_start_hands_the_chosen_files_and_thresholds_to_the_builder(self):
        # The Start path had never been exercised: it read `self.conf`/`self.iou`, which are not
        # attributes of this page, so pressing the button raised AttributeError.
        page = InferencerPage()
        page.data_config_path = "config.yaml"
        page.weights_path = "model.onnx"
        page.Conf.setValue(0.4)
        page.IoU.setValue(0.6)
        handed_over = {}

        def fake_build(**arguments):
            handed_over.update(arguments)
            return SimpleNamespace(
                engine=SimpleNamespace(output_names=(), preprocess=lambda frame: (frame, None)),
                output_config=None,
                input_config=SimpleNamespace(input_size=(640, 480)),
                class_names=(),
                skeleton=(),
                kpt_shape=(0, 3),
            )

        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.build_inference_session",
            side_effect=fake_build,
        ), patch.object(page, "_init_inference_threads"), patch.object(
            page, "_start_inference_threads"
        ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
            page.onStartClicked()

        self.assertEqual(handed_over["weights_path"], "model.onnx")
        self.assertEqual(handed_over["config_path"], "config.yaml")
        self.assertEqual(handed_over["conf"], 0.4)
        self.assertEqual(handed_over["iou"], 0.6)
        self.assertIsNotNone(page.session)
        self.assertFalse(page.Start.isEnabled())

    def test_start_reports_a_refused_artifact_instead_of_starting(self):
        page = InferencerPage()
        page.data_config_path = "config.yaml"
        page.weights_path = "simcc.onnx"

        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.build_inference_session",
            side_effect=ValueError("是 SimCCHead 模型"),
        ), patch.object(page, "_init_inference_threads") as threads, patch(
            "animalposetracker.gui.inferencerpage.QMessageBox"
        ) as box:
            page.onStartClicked()

        self.assertIn("无法启动推理", box.critical.call_args.args[2])
        threads.assert_not_called()
        self.assertIsNone(page.session)  # the run never began

    def test_a_video_source_supplies_the_frame_rate(self):
        # The video branch never set it: only the camera branch did, so the display timer later
        # divided by an attribute that was never written.
        page = self._page()
        page.fps = 0

        class Video(_FakeCapture):
            def get(self, prop):
                return 25.0

        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.cv2.VideoCapture", return_value=Video()
        ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
            page._init_inference_threads()

        self.assertEqual(page.fps, 25.0)

    def test_the_display_timer_survives_a_source_without_a_frame_rate(self):
        page = self._page()
        page.fps = 0
        page.display_thread = Mock()
        for thread in ("videoreader_thread", "preprocess_thread", "inference_thread",
                       "postprocess_thread", "visualize_thread"):
            setattr(page, thread, Mock())

        page._start_inference_threads()

        self.assertEqual(page.display_thread.start.call_args.args[0], 33)


class _FakeCapture:
    """Minimal stand-in for a camera/video handle."""

    def isOpened(self):
        return True

    def get(self, prop):
        return 0

    def set(self, prop, value):
        return True

    def release(self):
        return None


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class SubPageLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_opening_a_tool_window_builds_one_window_per_click(self):
        page = AnimalPoseTrackerPage()
        windows = [Mock(), Mock()]
        with patch("animalposetracker.gui.mainpage.WindowFactory.run", side_effect=windows) as run:
            page._open_sub_page(AnnotatorPage)
            page._open_sub_page(InferencerPage)

            self.assertEqual(run.call_count, 2)
            self.assertIs(page.sub_page, windows[1])

        # The window being replaced is closed and freed; it used to be one that had just been
        # built for the occasion and was thrown away unshown.
        windows[0].close.assert_called_once()
        windows[0].deleteLater.assert_called_once()


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class StatusBarEventTests(unittest.TestCase):
    """Evaluation results have to reach the status bar, not just the training ones."""

    EVALUATION = {
        "event": "evaluation",
        "metrics": {"coco/AP": 0.71, "coco/AP50": 0.88, "PCK": 0.62},
        "message": "Evaluation complete; metrics saved to runs/val/metrics.json",
    }

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_evaluation_event_shows_its_metrics(self):
        page = AnimalPoseTrackerPage()
        with patch.object(page.project, "read_training_events", return_value=[self.EVALUATION]):
            page._display_project_events()

        message = page.statusBar().currentMessage()
        self.assertIn("Evaluation complete", message)
        self.assertIn("pose mAP50-95=0.710", message)
        self.assertIn("PCK=0.620", message)

    def test_evaluation_mode_polls_the_event_stream(self):
        # Reading events only in training mode is why a finished evaluation used to show nothing.
        page = AnimalPoseTrackerPage()
        page.current_mode = "evaluate"
        page.project.process = Mock(poll=Mock(return_value=None))
        with patch.object(page.project, "read_training_events", return_value=[self.EVALUATION]):
            page._check_process()

        self.assertIn("pose mAP50-95=0.710", page.statusBar().currentMessage())

    def test_ultralytics_ap_namespace_is_shown_too(self):
        page = AnimalPoseTrackerPage()
        event = {
            "event": "evaluation",
            "metrics": {"ultralytics/AP": 0.42, "ultralytics/AP50": 0.6},
            "message": "Evaluation complete.",
        }
        with patch.object(page.project, "read_training_events", return_value=[event]):
            page._display_project_events()

        self.assertIn("pose mAP50-95=0.420", page.statusBar().currentMessage())

    def test_batch_loss_uses_the_per_image_figure_when_the_head_reports_one(self):
        # The SimCC head's `loss` is a batch sum: showing it beside the YOLO-Pose loss would put
        # two different scales under the same label, so the per-image figure wins.
        page = AnimalPoseTrackerPage()
        event = {
            "event": "batch",
            "epoch": 1,
            "epochs": 5,
            "step": 3,
            "steps": 10,
            "metrics": {"loss": 12.5, "loss_simcc_per_sample": 1.5625},
        }
        with patch.object(page.project, "read_training_events", return_value=[event]):
            page._display_project_events()

        message = page.statusBar().currentMessage()
        self.assertIn("loss=1.562", message)
        self.assertNotIn("12.5", message)

    def test_batch_loss_falls_back_to_the_plain_key(self):
        page = AnimalPoseTrackerPage()
        event = {
            "event": "batch", "epoch": 1, "epochs": 5, "step": 3, "steps": 10,
            "metrics": {"loss": 22.4},
        }
        with patch.object(page.project, "read_training_events", return_value=[event]):
            page._display_project_events()

        self.assertIn("loss=22.4", page.statusBar().currentMessage())


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class VideoWriterOutputTests(unittest.TestCase):
    """The saved video keeps the size of the frames it was handed.

    The page used to open the writer at the size typed into the width/height boxes, which is
    only what the source was asked for: a camera that answers with another size had every frame
    stretched to fit the declaration.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_output_keeps_the_size_of_the_frames_it_receives(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.avi"
            writer = VideoWriterThread(save_path=output, fps=5)
            writer.start()
            for index in range(4):
                writer.add_frame(np.full((24, 32, 3), index * 40, dtype=np.uint8))
            writer.stop_flag = True
            self.assertTrue(writer.wait(10000), "the writer thread did not finish")
            recorded = writer.frame_size

            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened(), output)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = 0
            while True:
                readable, _frame = capture.read()
                if not readable:
                    break
                frames += 1
            capture.release()

        self.assertEqual(recorded, (32, 24))
        self.assertEqual((width, height), (32, 24))
        self.assertEqual(frames, 4)
        self.assertEqual(writer.skipped_frames, 0)

    def test_a_size_change_mid_stream_is_counted_not_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.avi"
            writer = VideoWriterThread(save_path=output, fps=5)
            writer.start()
            writer.add_frame(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.add_frame(np.zeros((32, 48, 3), dtype=np.uint8))
            writer.stop_flag = True
            self.assertTrue(writer.wait(10000), "the writer thread did not finish")

            capture = cv2.VideoCapture(str(output))
            frames = 0
            while True:
                readable, _frame = capture.read()
                if not readable:
                    break
                frames += 1
            capture.release()

        self.assertEqual(frames, 1)
        self.assertEqual(writer.skipped_frames, 1)


if __name__ == "__main__":
    unittest.main()
