"""Headless checks for the PySide6 pages and the Qt threads they drive.

Offscreen Qt keeps these runnable without a display, and the module is skipped outright when
PySide6 is missing -- it ships as the optional ``gui`` extra, so the core suite must stay
green without it.
"""

import json
import os
import queue
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
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QLayout,
        QLineEdit,
        QMenu,
        QMenuBar,
        QScrollArea,
        QStackedWidget,
        QStyleOptionViewItem,
        QTabWidget,
        QWidget,
    )
    from PySide6.QtCore import Qt, QSize

    from animalposetracker.gui.annotatorpage import AnnotatorPage
    from animalposetracker.gui.createnewprojectpage import CreateNewProjectPage
    from animalposetracker.gui.inferencerpage import InferencerPage
    from animalposetracker.gui.mainpage import AnimalPoseTrackerPage, SplashScreen
    from animalposetracker.gui.publicdatasetsprojectpage import PublicDatasetProjectPage
    from animalposetracker.gui.utils import AnnotationTarget
    from animalposetracker.gui import WindowFactory
    from animalposetracker.utils import VideoReaderThread, VideoWriterThread
except ImportError:  # pragma: no cover - depends on the optional gui extra
    QApplication = None
    AnnotatorPage = None
    CreateNewProjectPage = None
    InferencerPage = None
    AnimalPoseTrackerPage = None
    PublicDatasetProjectPage = None
    AnnotationTarget = None
    WindowFactory = None
    SplashScreen = None
    VideoReaderThread = None
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

    def test_closing_the_form_stops_the_pending_debounce(self):
        # A pending debounce used to fire into the closed page.
        page = CreateNewProjectPage()
        page.ProjectConfig.setText("probe")
        page.onProjectTextEdited()
        self.assertTrue(page.edit_project_timer.isActive())

        self.assertTrue(page.close())

        self.assertFalse(page.edit_project_timer.isActive())

    def test_the_keypoint_rows_grow_and_clear_without_leftovers(self):
        # Typing into the last row appends a fresh empty one; clearing must take every extra row
        # (and its layout) back out, so the scroll area does not keep reporting rows the manager
        # no longer holds.
        page = CreateNewProjectPage()
        manager = page.keypoint_manager
        base = manager.v_layout.count()

        manager.input_rows[0]['input_field'].setText("head")
        self.assertEqual(manager.get_list_data(), ["head"])
        self.assertEqual(len(manager.input_rows), 2)
        self.assertEqual(manager.input_rows[-1]['input_field'].text(), "")
        self.assertEqual(manager.v_layout.count(), base + 1)

        manager.input_rows[-1]['input_field'].setText("tail")
        self.assertEqual(manager.get_list_data(), ["head", "tail"])

        manager.clear_inputs()
        self.assertEqual(manager.get_list_data(), [])
        self.assertEqual(len(manager.input_rows), 1)
        self.assertEqual(manager.v_layout.count(), base)


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
                page.onSelectPoserClicked()

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
            self.assertIn("class names", box.critical.call_args.args[2])

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
                page.onSelectPoserClicked()
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
                page.onSelectPoserClicked()

            self.assertFalse(page._weights_ready)
            self.assertFalse(page.Start.isEnabled())
            self.assertIn("predict", box.critical.call_args.args[2])
            # The check runs first, so the engine/device probe never happens.
            probe.assert_not_called()

    def test_loading_a_config_opens_both_model_buttons(self):
        # The config is the gate: before it, neither model button is usable; after it, both are.
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "dataset.yaml"
            config_path.write_text("names: {0: probe}\nkpt_shape: [2, 3]\n", encoding="utf-8")

            page = InferencerPage()
            self.assertFalse(page.SelectPoser.isEnabled())
            self.assertFalse(page.SelectDetector.isEnabled())

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ):
                page.onSelectConfigureClicked()

            self.assertTrue(page.SelectPoser.isEnabled())
            self.assertTrue(page.SelectDetector.isEnabled())

    def test_declared_weights_are_adopted_and_still_overridable(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "poser.onnx").write_bytes(b"")
            (base / "detector.onnx").write_bytes(b"")
            (base / "override.onnx").write_bytes(b"")
            config_path = base / "dataset.yaml"
            config_path.write_text(
                "names: {0: probe}\n"
                "kpt_shape: [2, 3]\n"
                "poser_weights: poser.onnx\n"
                "detector_weights: detector.onnx\n",
                encoding="utf-8",
            )

            page = InferencerPage()

            def fake_engine_detection(_extension):
                page.supported_engines_and_devices = {"ONNX": ["Intel CPU"]}

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"):
                page.onSelectConfigureClicked()

            # Relative paths resolve against the config file's directory.
            self.assertEqual(Path(page.weights_path), base / "poser.onnx")
            self.assertEqual(Path(page.detector_path), base / "detector.onnx")
            self.assertTrue(page._weights_ready)
            self.assertTrue(page.Start.isEnabled())

            # A later pick still overrides what the config declared.
            with patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(base / "override.onnx"), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"):
                page.onSelectPoserClicked()

            self.assertEqual(Path(page.weights_path), base / "override.onnx")

    def test_a_config_declaring_a_complete_simcc_pair_loads_cleanly(self):
        # The detector half is adopted first: adopting the SimCC poser before its detector
        # existed used to pop a refusal for a pair that is in fact complete.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "simcc.onnx").write_bytes(b"")
            (base / "simcc.onnx.json").write_text(
                '{"head": "SimCCHead", "kpt_shape": [2, 3]}', encoding="utf-8"
            )
            (base / "det.onnx").write_bytes(b"")
            config_path = base / "dataset.yaml"
            config_path.write_text(
                "names: {0: probe}\n"
                "kpt_shape: [2, 3]\n"
                "poser_weights: simcc.onnx\n"
                "detector_weights: det.onnx\n",
                encoding="utf-8",
            )

            page = InferencerPage()

            def fake_engine_detection(_extension):
                page.supported_engines_and_devices = {"ONNX": ["Intel CPU"]}

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"), patch(
                "animalposetracker.gui.inferencerpage.QMessageBox"
            ) as box:
                page.onSelectConfigureClicked()

            box.critical.assert_not_called()
            self.assertTrue(page._weights_ready)
            self.assertTrue(page._weights_is_simcc)
            self.assertEqual(Path(page.detector_path), base / "det.onnx")
            # One-stage refuses the SimCC half on its own; the switch makes the pair legal.
            self.assertFalse(page.Start.isEnabled())

            page.onModeToggled()
            self.assertTrue(page.Start.isEnabled())

    def test_the_detector_button_sits_before_the_poser_button(self):
        page = InferencerPage()
        layout = page.BaseModelsLayout

        self.assertLess(
            layout.indexOf(page.SelectDetector), layout.indexOf(page.SelectPoser)
        )

    def test_the_mode_switch_flips_the_stage_matrix(self):
        page = InferencerPage()
        self.assertEqual(page.ModeToggle.text(), "Mode: One-stage")

        page.tools_enabled()
        self.assertTrue(page.IoU.isEnabled())
        self.assertTrue(page.Conf.isEnabled())
        self.assertFalse(page.DetectorConf.isEnabled())

        page.onModeToggled()
        self.assertEqual(page.ModeToggle.text(), "Mode: Two-stage")
        self.assertFalse(page.IoU.isEnabled())
        self.assertFalse(page.Conf.isEnabled())
        self.assertTrue(page.DetectorConf.isEnabled())

    def test_two_stage_needs_a_simcc_pair_to_start(self):
        # Start tracks the stage pair, not just the two files: a SimCC artifact is refused
        # outright in one-stage, and in two-stage it stays dark until the detector arrives
        # -- the same rule the shared stream pipeline enforces in its constructor.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "best.onnx").write_bytes(b"")
            (base / "best.onnx.json").write_text(
                '{"head": "SimCCHead", "kpt_shape": [2, 3]}', encoding="utf-8"
            )
            (base / "det.onnx").write_bytes(b"")
            config_path = base / "dataset.yaml"
            config_path.write_text("names: {0: probe}\nkpt_shape: [2, 3]\n", encoding="utf-8")

            page = InferencerPage()

            def fake_engine_detection(_extension):
                page.supported_engines_and_devices = {"ONNX": ["Intel CPU"]}

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"):
                page.onSelectConfigureClicked()

            with patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(base / "best.onnx"), ""),
            ), patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"), patch(
                "animalposetracker.gui.inferencerpage.QMessageBox"
            ) as box:
                page.onSelectPoserClicked()

            # One-stage refuses the SimCC artifact (the refusal names the unlock).
            self.assertFalse(page._weights_ready)
            self.assertFalse(page.Start.isEnabled())
            self.assertIn("Two-stage", box.critical.call_args.args[2])

            # Two-stage without a detector is still not a runnable pair.
            page.onModeToggled()
            self.assertFalse(page.Start.isEnabled())

            with patch.object(
                page, "_get_supported_engines_and_devices", side_effect=fake_engine_detection
            ), patch.object(page, "_init_engine_selection"), patch.object(
                page, "_init_available_devices"
            ), patch.object(page, "_init_model_bits"):
                page._apply_detector_selection(str(base / "det.onnx"))

            # With the detector on hand the pair passes, and the pose half is revalidated
            # on the way in (it had been held back by the missing detector).
            self.assertTrue(page._weights_ready)
            self.assertTrue(page.Start.isEnabled())

    def test_the_live_thresholds_follow_their_boxes(self):
        # The decode stage reads plain attributes each frame; the boxes' handlers keep those
        # in step (widgets themselves are not touched from the worker threads).
        page = InferencerPage()
        page.Conf.setValue(0.4)
        page.IoU.setValue(0.5)
        page.DetectorConf.setValue(0.6)

        self.assertAlmostEqual(page._live_poser_conf, 0.4)
        self.assertAlmostEqual(page._live_nms_iou, 0.5)
        self.assertAlmostEqual(page._live_detector_conf, 0.6)

    def test_the_threshold_defaults_are_one_set(self):
        # The GUI used to open its IoU at 0.70 while the command line's deploy default was
        # 0.45; there is one set of numbers now.
        page = InferencerPage()
        self.assertAlmostEqual(page.IoU.value(), 0.45)
        self.assertAlmostEqual(page.Conf.value(), 0.25)
        self.assertAlmostEqual(page.DetectorConf.value(), 0.5)


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class ConfigureTreeTests(unittest.TestCase):
    """The parameter trees must only offer keys the backend actually reads.

    A row whose key is not in the project's runtime.yaml raises the moment it is edited, and a
    row whose key nothing consumes is a knob that silently does nothing. The trees are generated
    from the key registry's `views` annotations, so only registered keys reach the screen.
    """

    # The exact rows each tree should end up with. Editing this list is the point: it forces
    # every addition or removal to be a deliberate decision. This is the startup set: with no
    # project loaded the model-scoped rows all show, so the page previews the whole registry
    # (loading a project narrows the train form to that model's recipe -- see
    # `test_model_scoped_rows_follow_the_active_model`).
    EXPECTED = {
        "TrainingConfigureEdit": {
            "model", "data", "num_workers", "time_hours", "cache", "fraction", "profile",
            "plots", "validation_enabled", "save", "save_period", "pretrained", "resume",
            "early_stopping_patience", "epochs", "batch_size", "image_size", "optimizer",
            "learning_rate", "final_lr_factor", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "nominal_batch_size", "cosine_schedule",
            "amp", "close_mosaic", "seed", "box_loss_weight", "class_loss_weight",
            "distribution_loss_weight", "keypoint_loss_weight", "visibility_loss_weight",
            "oks_loss_weight", "mle_loss_weight", "overlaps_power",
            "device", "project", "name",
        },
        "EvaluateConfigure": {
            "data", "num_workers", "profile", "plots", "verbose", "single_cls",
            "batch_size", "image_size", "confidence", "iou", "max_det", "agnostic_nms",
            "device", "project", "name",
        },
        "InferenceConfigure": {
            "save", "verbose", "batch_size", "image_size", "device", "project", "name",
            "source", "confidence", "iou", "max_det", "agnostic_nms", "show_conf",
            "show_boxes", "show_skeletons", "line_width", "point_radius",
            # Grouped rows keep their group-qualified names (the tracking / temporal-filter
            # blocks in runtime.yaml).
            "tracking.enabled", "tracking.algorithm",
            "pose_filter.method", "pose_filter.window_length", "pose_filter.polyorder",
            "pose_filter.max_gap",
        },
        # Smaller than the others: only the keys flagged for the export view.
        "ExportConfigure": {
            "data", "fraction", "batch_size", "image_size", "device",
            "format", "half", "dynamic", "simplify", "opset", "workspace", "optimize",
            "int8", "keras", "nms",
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

    def test_model_scoped_rows_follow_the_active_model(self):
        # Rows whose group is model-scoped only belong on the form of the model whose
        # recipe defines them: a YOLO-pipeline loss weight means nothing to the SimCC
        # pipeline, and the one-stage head's own loss terms mean nothing to the others.
        cases = {
            "AnimalRTPose": {
                "present": {"distribution_loss_weight", "keypoint_loss_weight", "box_loss_weight"},
                "absent": {"oks_loss_weight", "mle_loss_weight", "overlaps_power"},
            },
            "AnimalViTPose": {
                "present": {"seed"},  # model-scoped, but also registered under shared.runtime
                "absent": {"distribution_loss_weight", "keypoint_loss_weight", "box_loss_weight"},
            },
            "AnimalRTMO": {
                "present": {"oks_loss_weight", "mle_loss_weight", "overlaps_power", "box_loss_weight"},
                "absent": {"distribution_loss_weight", "keypoint_loss_weight"},
            },
        }
        for model_type, expectation in cases.items():
            page = AnimalPoseTrackerPage()
            page.project.project_config["model_type"] = model_type
            # A loaded project carries its runtime configuration; that is what tells the
            # page which model's rows to show.
            page.project.runtime_config["common"] = {"device": "auto"}
            page.initialize_other_config()
            keys = self._keys(page.TrainingConfigureEdit)
            with self.subTest(model=model_type):
                self.assertTrue(expectation["present"] <= keys)
                self.assertFalse(expectation["absent"] & keys)
                # Every form keeps the shared rows.
                self.assertIn("epochs", keys)
                self.assertIn("batch_size", keys)
                self.assertIn("device", keys)

    def test_prediction_thresholds_land_in_the_prediction_section(self):
        # The inference page's `confidence`/`iou` rows are prediction thresholds: they must
        # resolve into the predict section, while the evaluate page keeps its validation pair.
        page = AnimalPoseTrackerPage()
        self.assertEqual(
            page._runtime_path_for_gui_key("confidence", "InferenceConfigure"), "predict.confidence")
        self.assertEqual(
            page._runtime_path_for_gui_key("iou", "InferenceConfigure"), "predict.iou")
        self.assertEqual(
            page._runtime_path_for_gui_key("confidence", "EvaluateConfigure"),
            "train.models.AnimalRTPose.validation.confidence")
        self.assertEqual(
            page._runtime_path_for_gui_key("iou", "EvaluateConfigure"),
            "train.models.AnimalRTPose.validation.iou")

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
        # runtime.yaml holds tracking and pose_filter as mapping blocks, so the rows
        # address one field of each; writing the row has to land inside the mapping
        # the pipeline reads.
        page = AnimalPoseTrackerPage()
        page.project.runtime_config["predict"] = {
            "tracking": {"enabled": False, "algorithm": "bytetrack"},
            "pose_filter": {"method": "none", "window_length": 5},
        }
        page.initialize_other_config()

        tree = page.InferenceConfigure
        self.assertEqual(self._row(tree, "tracking.enabled").text(1), "False")
        self._row(tree, "tracking.enabled").setText(1, "True")
        self._row(tree, "pose_filter.window_length").setText(1, "9")

        self.assertIs(page.project.runtime_config["predict"]["tracking"]["enabled"], True)
        self.assertEqual(
            page.project.runtime_config["predict"]["pose_filter"]["window_length"], 9
        )

    def test_the_nested_rows_are_editable_from_the_runtime_snapshot(self):
        # tracking / pose_filter ship in the runtime snapshot, so their rows are
        # editable on a fresh project and an edit lands inside the mapping.
        page = AnimalPoseTrackerPage()
        page.project.runtime_config["predict"] = {
            "tracking": {"enabled": False, "algorithm": "bytetrack"},
            "pose_filter": {"method": "none"},
        }
        page.initialize_other_config()

        self.assertEqual(page.project.runtime_config["predict"]["tracking"]["enabled"], False)

        self._row(page.InferenceConfigure, "tracking.enabled").setText(1, "True")
        self.assertIs(page.project.runtime_config["predict"]["tracking"]["enabled"], True)
        # A row whose mapping the tree lacks stays locked instead of pretending.
        page.project.runtime_config["predict"].pop("tracking")
        page.initialize_other_config()
        self.assertFalse(
            bool(
                self._row(page.InferenceConfigure, "tracking.enabled").flags()
                & Qt.ItemIsEditable
            )
        )

    def test_a_row_the_project_lacks_is_cleared_and_locked(self):
        # Switching projects: a row the newly opened project does not carry used to keep the
        # previous project's value and stay editable, while an edit to it was discarded on save.
        page = AnimalPoseTrackerPage()
        training = {"models": {"AnimalRTPose": {"training": {"image_size": 640}}}}
        page.project.runtime_config["train"] = training
        page.initialize_other_config()
        row = self._row(page.InferenceConfigure, "image_size")
        self.assertEqual(row.text(1), "640")
        self.assertTrue(bool(row.flags() & Qt.ItemIsEditable))

        training["models"]["AnimalRTPose"]["training"].pop("image_size")
        page.initialize_other_config()
        row = self._row(page.InferenceConfigure, "image_size")
        self.assertEqual(row.text(1), "")
        self.assertFalse(bool(row.flags() & Qt.ItemIsEditable))

        training["models"]["AnimalRTPose"]["training"]["image_size"] = 512
        page.initialize_other_config()
        row = self._row(page.InferenceConfigure, "image_size")
        self.assertEqual(row.text(1), "512")
        self.assertTrue(bool(row.flags() & Qt.ItemIsEditable))

    def test_edit_mode_keeps_a_row_the_project_lacks_locked(self):
        # Edit mode used to unlock every row, including the ones an edit to which is dropped.
        page = AnimalPoseTrackerPage()
        # `image_size` is carried by the train profile -> editable; `save` lives in a
        # section this tree does not carry -> stays locked.
        page.project.runtime_config["train"] = {
            "models": {"AnimalRTPose": {"training": {"image_size": 640}}}
        }
        page.initialize_other_config()
        page.EditInferenceParameters.click()  # "Edit ..." -> edit mode
        self.assertTrue(
            bool(self._row(page.InferenceConfigure, "image_size").flags() & Qt.ItemIsEditable)
        )
        self.assertFalse(
            bool(self._row(page.InferenceConfigure, "save").flags() & Qt.ItemIsEditable)
        )

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
        # `cache` accepts two words besides a bool, so it is listed rather than derived.
        self.assertEqual(page.configure_choices("cache"), ("True", "False", "ram", "disk"))
        # Numbers and paths stay free text.
        self.assertEqual(page.configure_choices("epochs"), ())
        self.assertEqual(page.configure_choices("source"), ())

    def test_boolean_rows_are_recognised_from_the_config_value(self):
        page = AnimalPoseTrackerPage()
        page.project.runtime_config["predict"] = {"tracking": {"enabled": False}}
        page.project.runtime_config["train"] = {
            "models": {"AnimalRTPose": {"training": {"epochs": 10}}}
        }

        self.assertEqual(page.configure_choices("tracking.enabled"), ("True", "False"))
        self.assertEqual(page.configure_choices("epochs"), ())

    def test_rows_that_also_take_text_keep_the_text_editor(self):
        # `pretrained` names a checkpoint when it holds a string, so a True/False dropdown would
        # take away a capability rather than add one.
        page = AnimalPoseTrackerPage()
        self.assertEqual(page.configure_choices("pretrained"), ())

    def test_the_value_column_edits_with_a_dropdown_when_there_are_choices(self):
        page = AnimalPoseTrackerPage()
        page.project.runtime_config["predict"] = {
            "tracking": {"enabled": True, "algorithm": "bytetrack"}
        }
        page.initialize_other_config()
        tree = page.InferenceConfigure

        selected = self._editor(page, tree, "tracking.enabled")
        self.assertIsInstance(selected, QComboBox)
        self.assertEqual(
            [selected.itemText(index) for index in range(selected.count())], ["True", "False"]
        )
        # A row with no fixed set still opens the plain text editor.
        self.assertIsInstance(self._editor(page, tree, "image_size"), QLineEdit)


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

    def test_the_two_stage_pipeline_rewires_the_middle_stages(self):
        # With the shared stream pipeline in place the frame reaches it whole: the detector
        # does the cutting, and the pipeline decodes on the way through, so the other two
        # stages become pass-throughs instead of the engine's preprocess + the page's decode.
        page = self._page()
        page._stream_pipeline = SimpleNamespace()
        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.cv2.VideoCapture", return_value=_FakeCapture()
        ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
            page._init_inference_threads()

        self.assertEqual(page.preprocess_thread.preprocess_function, page._pass_frame_through)
        self.assertEqual(page.inference_thread.inference_function, page._infer_topdown)
        self.assertEqual(page.postprocess_thread.postprocess_function, page._wrap_instances)

    def test_start_in_two_stage_builds_the_shared_stream_pipeline(self):
        # Two-stage tells the builder a detector exists -- which is what unlocks the SimCC
        # schema -- and assembles the same detect -> crop -> decode pipeline the command
        # line runs, with the conventions read off the detector file's name.
        page = InferencerPage()
        page.data_config_path = "config.yaml"
        page.weights_path = "simcc.onnx"
        page.detector_path = "yolo-det.onnx"
        page._two_stage = True
        handed_over, detector_args, pipeline_args = {}, {}, {}

        def fake_build(**arguments):
            handed_over.update(arguments)
            return SimpleNamespace(
                engine=object(),
                output_config=object(),
                input_config=SimpleNamespace(input_size=(640, 480)),
                class_names=("mouse",),
                skeleton=(),
                kpt_shape=(2, 3),
            )

        class FakeDetector:
            def __init__(self, path, **kwargs):
                detector_args.update({"path": path, **kwargs})

        class FakePipeline:
            def __init__(self, engine, output_config, **kwargs):
                pipeline_args.update(
                    {"engine": engine, "output_config": output_config, **kwargs}
                )

        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.build_inference_session", side_effect=fake_build
        ), patch(
            "animalposetracker.inference.detector_adapter.OnnxDetectorAdapter", FakeDetector
        ), patch(
            "animalposetracker.inference.stream_inference.StreamInferencePipeline", FakePipeline
        ), patch.object(page, "_init_inference_threads"), patch.object(
            page, "_start_inference_threads"
        ), patch("animalposetracker.gui.inferencerpage.QMessageBox"):
            page.onStartClicked()

        self.assertIs(handed_over["topdown"], True)
        self.assertEqual(pipeline_args["input_mode"], "topdown")
        self.assertEqual(pipeline_args["class_names"], ["mouse"])
        # A yolo* detector file speaks pixels + letterbox + xyxy.
        self.assertEqual(detector_args["box_format"], "xyxy")
        self.assertEqual(detector_args["coord_mode"], "pixels")
        self.assertEqual(detector_args["preprocess_mode"], "letterbox")
        self.assertIsNotNone(page._stream_pipeline)

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

    def test_every_frame_is_collected_as_a_record(self):
        # The records are the run's result, so every frame is kept and written out when the run
        # ends -- there is no switch to ask for them first.
        page = self._page()
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

    def test_the_one_save_switch_joins_the_start_stop_allow_lists(self):
        page = self._page()

        page.tools_disabled()
        self.assertFalse(page.Save.isEnabled())
        page.tools_enabled()
        self.assertTrue(page.Save.isEnabled())

    def test_the_records_file_is_written_without_being_asked_for(self):
        page = self._page()
        page.records = [{"frame": 0, "predictions": []}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            Path, "cwd", return_value=Path(temporary)
        ):
            page._finalize_cleanup()

            target = Path(temporary) / "runs" / "inference" / "records.json"
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), page.records)

    def test_a_run_that_produced_no_records_writes_no_file(self):
        page = self._page()
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            Path, "cwd", return_value=Path(temporary)
        ):
            page._finalize_cleanup()

            self.assertFalse((Path(temporary) / "runs" / "inference" / "records.json").exists())

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
        page.FilterMethod.setCurrentText("median")
        self._feed(page)

        # The switch is checked, but there are no track ids for the filter to work on.
        self.assertFalse(page._filter_is_on())
        self.assertIs(page._final_records(), page.records)

    def test_the_layout_follows_the_third_plan(self):
        page = self._page()
        left = page.BaseToolsLayout

        # Row 1 is the setup triple: configure, the mode switch and the source picker; row 2
        # is the model pair -- detector first, poser second.
        self.assertIs(left.itemAt(0).layout(), page.BaseConfigureLayout)
        self.assertEqual(
            [left.itemAt(0).layout().itemAt(index).widget() for index in range(3)],
            [page.SelectConfigure, page.ModeToggle, page.CameraORVideos],
        )
        self.assertIs(left.itemAt(1).layout(), page.BaseModelsLayout)
        # Rows 3 and 4 hold the two optional stages and the filter's values. The tracker's row is
        # the empty TrackerLayout the ui module leaves in the engine/device block, moved up and
        # filled -- not a second layout wearing the same name.
        self.assertIs(left.itemAt(2).layout(), page.TrackerLayout)
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
        # Tracking moved out of that row, leaving the three pairs the ui module drew.
        self.assertEqual(
            [page.EngineDeviceLayout.itemAt(index).layout() for index in range(3)],
            [page.ModelBitsLayout, page.EngineLayout, page.DeviceLayout],
        )

        # The display group opens with the decode trio -- the boxes that keep reacting while a
        # run is going -- and their widgets stay inside that group.
        right = page.ShowToolsGroupVLayout
        self.assertIs(right.itemAt(0).layout(), page.DecodeParamsLayout)
        self.assertEqual(page.IoU.parentWidget().objectName(), "ShowToolsGroup")
        self.assertEqual(page.Conf.parentWidget().objectName(), "ShowToolsGroup")
        self.assertEqual(page.DetectorConf.parentWidget().objectName(), "ShowToolsGroup")
        # Then the switch that writes the video, in the row that lists what the run shows -- it
        # had a row of its own while there was a records switch to pair it with.
        self.assertIs(right.itemAt(1).layout(), page.MainShowLayout)
        self.assertEqual(
            [right.itemAt(1).layout().itemAt(index).widget() for index in range(4)],
            [page.Save, page.Show, page.Backgroud, page.ShowClasses],
        )
        # Seven rows: the decode trio, the two switch rows and the four the sliders sit in.
        self.assertEqual(right.count(), 7)

    def test_the_rows_fill_their_group_the_way_the_form_lays_them_out(self):
        # The ui module gives this group's controls an expanding width, so a row divides its
        # group between its cells -- that is the look the page has always had, and the one to keep.
        # A pass that clamped every control to one cell width and padded the row instead shrank the
        # controls to their text and left the right half of each row empty.
        page = self._page()
        page.resize(1194, 800)
        page.layout().activate()
        QApplication.processEvents()

        for attribute in (
            "SelectConfigure", "ModeToggle", "SelectDetector", "SelectPoser",
            "WidthSetup", "HeightSetup", "FPSSetup",
            "ModelBits", "EngineSelection", "DeviceSelection", "Start", "End",
            "IoU", "Conf", "DetectorConf",
            "CameraVideosSelection", "TrackerAlgorithm", "FilterMethod", "FilterWindow",
            "FilterConfidence",
        ):
            with self.subTest(widget=attribute):
                # QWIDGETSIZE_MAX: no upper bound of the page's own.
                self.assertEqual(getattr(page, attribute).maximumWidth(), 16777215)

        # A trailing stretch would pack a row against the left edge. No row has one, the rows the
        # ui module drew included: they divide the group, they do not hug its left side.
        for layout in (
            page.BaseConfigureLayout, page.BaseModelsLayout, page.DecodeParamsLayout,
            page.TrackerLayout,
            page.FilterValuesLayout, page.CameraVideosLayout, page.CameraParametersLayout,
            page.EngineDeviceLayout, page.StartEndLayout, page.MainShowLayout,
            page.ShowInfoLayput, page.FontScaleLayout,
            page.ShowKeypointsLayout, page.ShowSkeletonsLayout, page.ShowBBoxLayout,
        ):
            with self.subTest(layout=layout.objectName()):
                self.assertIsNone(layout.itemAt(layout.count() - 1).spacerItem())

        # Which shows up as cells of equal width: the setup row splits three ways, the model
        # pair two ways, and the four filter values four ways.
        for layout, count in (
            (page.BaseConfigureLayout, 3),
            (page.BaseModelsLayout, 2),
            (page.FilterValuesLayout, 4),
        ):
            widths = [layout.itemAt(index).widget().width() for index in range(count)]
            with self.subTest(layout=layout.objectName()):
                self.assertLessEqual(max(widths) - min(widths), 2)

    def test_the_two_switch_rows_start_from_the_same_cells(self):
        # The display group keeps one column rhythm: Save opens the first row at the module's
        # 120, and the FPS switch opens the second row the same way, so both rows start their
        # second cell at the same x and split what is left between three cells each.
        page = self._page()
        page.resize(1194, 800)
        page.layout().activate()
        QApplication.processEvents()

        self.assertEqual(page.Save.width(), 120)
        self.assertEqual(page.FPSShow.width(), 120)
        self.assertEqual(page.FPSShow.x(), page.Save.x())
        self.assertEqual(page.PreprocessTime.x(), page.Show.x())

    def test_the_left_group_labels_stand_in_one_column(self):
        # Rows read as a table when the labels are all one width: the field column then starts at
        # the same x in every row. Left to their own text, the fields began at 60 in the threshold
        # row, 74 in the camera row and 101 in the engine row.
        page = self._page()
        page.resize(1194, 800)
        page.layout().activate()
        QApplication.processEvents()

        labels = (
            page.WidthLabel, page.HeightLabel, page.FPSLabel,
            page.ModelBitsLabel, page.EngineLabel, page.DeviceLabel,
        )
        widest = max(label.sizeHint().width() for label in labels)
        for label in labels:
            with self.subTest(label=label.objectName()):
                self.assertEqual(label.minimumWidth(), widest)
                self.assertEqual(label.maximumWidth(), widest)

        # Which puts the fields of the two three-cell rows in the same columns, of one width.
        self.assertEqual(
            [page.WidthSetup.x(), page.HeightSetup.x(), page.FPSSetup.x()],
            [page.ModelBits.x(), page.EngineSelection.x(), page.DeviceSelection.x()],
        )
        self.assertEqual(
            [page.WidthSetup.width(), page.HeightSetup.width(), page.FPSSetup.width()],
            [page.ModelBits.width(), page.EngineSelection.width(), page.DeviceSelection.width()],
        )
        # The decode trio instead aligns among itself, over in the display column.
        trio = (page.IoULabel, page.DetectorConfLabel, page.ConfLabel)
        widest_trio = max(label.sizeHint().width() for label in trio)
        for label in trio:
            with self.subTest(label=label.objectName()):
                self.assertEqual(label.minimumWidth(), widest_trio)
                self.assertEqual(label.maximumWidth(), widest_trio)

    def test_no_widget_or_row_is_left_out_of_the_page(self):
        # A widget or a row that sits in no layout draws nothing, and in the source it reads as
        # something that was meant to be mounted and never was. Two rows had been left that way:
        # the row the thresholds moved out of, and a second TrackerLayout that took over the
        # module's slot without ever being added to a row itself.
        page = self._page()
        reachable_layouts, reachable_widgets = set(), set()

        def walk(layout):
            reachable_layouts.add(id(layout))
            for index in range(layout.count()):
                item = layout.itemAt(index)
                widget = item.widget()
                if widget is not None:
                    reachable_widgets.add(id(widget))
                    if widget.layout() is not None:
                        walk(widget.layout())
                elif item.layout() is not None:
                    walk(item.layout())

        walk(page.layout())

        left_out = [
            name
            for name, value in sorted(vars(page).items())
            if isinstance(value, QLayout) and id(value) not in reachable_layouts
        ] + [
            name
            for name, value in sorted(vars(page).items())
            if isinstance(value, QWidget) and id(value) not in reachable_widgets
        ]
        self.assertEqual(left_out, [])

    def test_the_save_switch_says_what_it_saves(self):
        # "Save" next to the other switches left the reader guessing what was being saved.
        page = self._page()

        self.assertEqual(page.Save.text(), "Save Video")
        self.assertIn("output.avi", page.Save.toolTip())
        # The records have no switch of their own any more, so the tooltip is where it is said.
        self.assertIn("records.json", page.Save.toolTip())

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
            side_effect=ValueError("is a SimCCHead model"),
        ), patch.object(page, "_init_inference_threads") as threads, patch(
            "animalposetracker.gui.inferencerpage.QMessageBox"
        ) as box:
            page.onStartClicked()

        self.assertIn("Cannot start inference", box.critical.call_args.args[2])
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
        "metrics": {"COCO/AP": 0.71, "COCO/AP50": 0.88, "PCK": 0.62},
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

    def test_yolo_ap_namespace_is_shown_too(self):
        page = AnimalPoseTrackerPage()
        event = {
            "event": "evaluation",
            "metrics": {"YOLO/AP": 0.42, "YOLO/AP50": 0.6},
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
class FrameExtractionTests(unittest.TestCase):
    """Extraction has to run on a worker; run inline, the click froze the window."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_extraction_runs_on_a_worker_and_the_button_comes_back(self):
        page = AnimalPoseTrackerPage()
        calls = []

        class FakeExtractor:
            output_dir = None

            def extract_keyframes_from_videos(self, **kwargs):
                calls.append(("videos", kwargs))

            def extract_keyframes_from_images(self, **kwargs):
                calls.append(("images", kwargs))

        page.extractor = FakeExtractor()
        page.project.project_config["sources"] = ["C:/does/not/exist/clip.mp4"]
        page.ExtractionMethodSelection.setCurrentText("auto")
        page.onExtractFrames()

        # The click returned with the run handed to a worker, and the button stays locked until
        # the worker reports back.
        self.assertIsNotNone(page._extract_thread)
        self.assertFalse(page.ExtracFrames.isEnabled())

        self.assertTrue(page._extract_thread.wait(5000), "the extraction worker did not finish")
        QApplication.processEvents()  # deliver the worker's `finished` to the page
        self.assertEqual([kind for kind, _ in calls], ["videos"])
        self.assertTrue(page.ExtracFrames.isEnabled())
        self.assertEqual(page.statusBar().currentMessage(), "Frame extraction complete.")


class _FramesThenStop:
    """A capture that yields a fixed list of frames and then reports the end of stream."""

    def __init__(self, frames):
        self._frames = list(frames)

    def isOpened(self):
        return True

    def get(self, prop):
        return 0  # a camera that has not started streaming yet

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        return None


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class InferenceWorkerContractTests(unittest.TestCase):
    """Queue, shutdown and reader contracts behind the inferencer page."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_a_full_queue_does_not_block_the_page(self):
        # The put helpers used the blocking `put()`, so one dead consumer froze the GUI thread
        # behind an `except queue.Full` that could never fire.
        page = InferencerPage()
        page.read_cache = queue.Queue(maxsize=1)
        page.read_cache.put("first")

        page.put_read_data("second")  # must return, not block

        self.assertEqual(page.read_cache.qsize(), 1)

    def test_end_with_save_checked_stops_the_writer_and_resets_the_page(self):
        page = InferencerPage()
        page.Save.setChecked(True)
        writer = Mock()
        writer.isRunning.return_value = True
        page.videowriter_thread = writer
        page._run_active = True

        page.onEndClicked()

        self.assertTrue(writer.stop_flag)
        writer.safe_stop.assert_called_once()
        self.assertFalse(page._run_active)
        self.assertEqual(page.Start.text(), "Start")

    def test_the_reader_survives_a_zero_fps_source(self):
        # 1 / 0 used to end the reader after its first frame: a preview or a run that stopped on
        # its own while the camera reported 0 fps before it started streaming.
        reader = VideoReaderThread()
        reader.cap = _FramesThenStop([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)])
        received = []
        reader.data_ready.connect(received.append)

        reader.start()
        self.assertTrue(reader.wait(10000), "the reader did not finish")
        QApplication.processEvents()  # deliver the queued `data_ready` emissions

        self.assertEqual(len(received), 3)
        self.assertFalse(reader.is_running)


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class AppShellTests(unittest.TestCase):
    """The splash fallback and what the tool-window factory hands a new window."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_the_splash_falls_back_to_a_blank_pixmap_for_a_missing_image(self):
        splash = SplashScreen("C:/does/not/exist/welcome.png")
        try:
            self.assertFalse(splash.pixmap().isNull())
            self.assertEqual(splash.pixmap().size(), QSize(800, 600))
        finally:
            splash.close()

    def test_the_factory_gives_a_tool_window_its_theme_and_lifetime(self):
        window = WindowFactory.run(
            QWidget, delete_on_close=True, stylesheet="QWidget { color: red; }"
        )
        try:
            self.assertEqual(window.styleSheet(), "QWidget { color: red; }")
            self.assertTrue(window.testAttribute(Qt.WA_DeleteOnClose))
        finally:
            window.close()


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


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class ConfigEditorFidelityTests(unittest.TestCase):
    """Opening a project's config must not change it.

    `AddItemsToTree` labels each row's path with `setData`, which emits `itemChanged` -- the very
    signal the editor slot listens to. Merely displaying a project therefore ran every value
    through `convert_value`: `'0012'` became `12`, `''` became `None`. Saving after a look wrote
    those types to disk, and the project could not be loaded again.
    """

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def _page_with_config(self, config):
        page = AnimalPoseTrackerPage()
        page.project.project_config = dict(config)
        page.config_data = page.project.project_config
        return page

    def test_displaying_the_tree_does_not_rewrite_the_config(self):
        original = {
            "project_name": "0012",
            "worker": "1.0",
            "date": "20240101",
            "note": "",
            "classes_name": ["1", "2"],
        }
        page = self._page_with_config(original)

        page.DisplayConfigureFile()

        self.assertEqual(page.project.project_config, original)

    def test_editing_a_row_still_reaches_the_config(self):
        # The other half of the contract: display must be inert, but a real edit must still land.
        page = self._page_with_config({"project_name": "old", "worker": "wux"})
        page.DisplayConfigureFile()

        item = page.ConfigureFile.topLevelItem(0)
        self.assertEqual(item.text(0), "project_name")
        item.setText(1, "new")
        page.OnConfigureFileEdited(item, 1)

        self.assertEqual(page.project.project_config["project_name"], "new")

    def test_clearing_a_cell_does_not_write_null(self):
        page = self._page_with_config({"project_name": "keep", "worker": "wux"})
        page.DisplayConfigureFile()

        item = page.ConfigureFile.topLevelItem(0)
        item.setText(1, "")
        page.OnConfigureFileEdited(item, 1)

        # `None` here made `"-".join(components)` fail the next time the path was rebuilt.
        self.assertEqual(page.project.project_config["project_name"], "")

    def test_clear_videos_images_empties_the_sources(self):
        page = self._page_with_config({"sources": ["a.mp4"]})

        page.onClearVideosImages()

        self.assertEqual(page.project.project_config["sources"], [])


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class TrainingMonitorTests(unittest.TestCase):
    """The poll timer and the exit-code contract of the training worker."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _finished_process(returncode):
        class _Process:
            def __init__(self, code):
                self.returncode = code

            def poll(self):
                return self.returncode

        return _Process(returncode)

    def test_a_missing_process_stops_the_timer_instead_of_raising(self):
        page = AnimalPoseTrackerPage()
        page.current_mode = "train"
        page._start_check_thread()
        self.assertTrue(page.check_thread.isActive())

        page.project.process = None
        page._check_process()  # used to raise, leaving a timer that threw once a second

        self.assertIsNone(page.check_thread)

    def test_ending_before_starting_does_not_raise(self):
        page = AnimalPoseTrackerPage()

        page.onEndEvaluate()
        page.onEndInference()

        self.assertIsNone(page.check_thread)

    def test_a_normal_finish_says_nothing_and_restores_start(self):
        page = AnimalPoseTrackerPage()
        page.current_mode = "train"
        page._start_check_thread()
        page.project.process = self._finished_process(0)

        page._check_process()

        self.assertEqual(page.statusBar().currentMessage(), "")
        self.assertTrue(page.StartTrain.isEnabled())
        self.assertIsNone(page.check_thread)

    def test_an_external_stop_is_reported(self):
        page = AnimalPoseTrackerPage()
        page.current_mode = "train"
        page._start_check_thread()
        page.project.process = self._finished_process(130)

        page._check_process()

        self.assertIn("stopped", page.statusBar().currentMessage())

    def test_an_error_is_reported_with_its_code(self):
        page = AnimalPoseTrackerPage()
        page.current_mode = "train"
        page._start_check_thread()
        page.project.process = self._finished_process(1)

        page._check_process()

        self.assertIn("1", page.statusBar().currentMessage())


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class InferencerStartContractTests(unittest.TestCase):
    """The engine vocabulary, the queue helpers and the records file."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_every_engine_label_maps_to_a_builder_key(self):
        from animalposetracker.inference.session import ENGINE_NAMES

        page = InferencerPage()
        for label in ENGINE_NAMES.values():
            with self.subTest(label=label):
                page.EngineSelection.clear()
                page.EngineSelection.addItem(label)
                page.EngineSelection.setCurrentText(label)

                # Handing the display label to the builder raised `KeyError: 'ONNX'`, so no GUI
                # run ever started.
                self.assertIn(page._engine_key(), ENGINE_NAMES)

    def test_a_full_queue_does_not_block_the_caller(self):
        page = InferencerPage()
        page.read_cache = queue.Queue(maxsize=1)
        page.read_cache.put("first")

        # Blocking `put()` never raises `Full`, so the handler was dead code and the GUI thread
        # waited forever once a consumer died.
        page.put_read_data("second")

        self.assertEqual(page.read_cache.qsize(), 1)

    def test_the_records_survive_the_tool_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = InferencerPage()
            page.records = [{"frame": 0, "predictions": []}]
            with patch.object(Path, "cwd", return_value=Path(temporary)):
                # What `mainpage._open_sub_page` does when the user picks another tool: the write
                # used to be deferred behind a timer that ran after the page was destroyed.
                page.close()
                page.deleteLater()
                self.application.processEvents()

                target = Path(temporary) / "runs" / "inference" / "records.json"
                self.assertTrue(target.exists(), "the run's records were lost")
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), page.records)

    def test_start_stays_dark_without_a_config_and_weights(self):
        page = InferencerPage()

        with patch.object(page, "_select_video_file"):
            page.tools_enabled()

        self.assertFalse(page.Start.isEnabled())

    def test_a_second_start_is_refused_while_a_run_is_active(self):
        page = InferencerPage()
        page._run_active = True
        page.data_config_path = "config.yaml"
        page.weights_path = "model.onnx"

        with patch.object(page, "_get_source", return_value="video.mp4"), patch(
            "animalposetracker.gui.inferencerpage.build_inference_session"
        ) as builder:
            page.onStartClicked()

        builder.assert_not_called()


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class NewProjectSafetyTests(unittest.TestCase):
    """The create-project form: validation, path fidelity, and what it leaves behind."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def _page(self, temporary, name="probe", worker="wux", keypoints=6, classes=2):
        page = CreateNewProjectPage()
        page.ProjectConfig.setText(name)
        page.WorkerConfig.setText(worker)
        page.LocationPathDisplay.setText(temporary)
        page.KeypointConfig.setValue(keypoints)
        page.ClassConfig.setValue(classes)
        page.onEditProjectTimerTimeout()
        page.onWorkerProjectTimerTimeout()
        return page

    def test_the_default_form_explains_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = self._page(temporary, keypoints=0, classes=0)
            with patch("animalposetracker.gui.createnewprojectpage.QMessageBox") as box:
                # `[1.0 / value] * value` raised ZeroDivisionError on the form's default state,
                # so Create did nothing at all and said nothing.
                page.onCreateProjectClicked()

            self.assertTrue(box.warning.called)

    def test_the_recorded_path_matches_the_created_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = self._page(temporary)
            page.onCreateProjectClicked()

            recorded = yaml.safe_load(
                (page.project.project_path / "project.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(recorded["project_path"], str(page.project.project_path))

    def test_an_existing_project_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = self._page(temporary, name="dup")
            first.onCreateProjectClicked()
            config = first.project.project_path / "project.yaml"
            config.write_text("# SENTINEL\n" + config.read_text(encoding="utf-8"), encoding="utf-8")

            second = self._page(temporary, name="dup")
            with patch("animalposetracker.gui.createnewprojectpage.QMessageBox"):
                second.onCreateProjectClicked()

            self.assertIn("SENTINEL", config.read_text(encoding="utf-8"))

    def test_a_failed_create_leaves_no_partial_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "dest"
            destination.mkdir()
            page = self._page(str(destination), name="half")
            page.project.update_config("project", {"sources": [str(Path(temporary) / "gone")]})
            page.CopySourceData.setChecked(True)

            with patch("animalposetracker.gui.createnewprojectpage.QMessageBox"):
                page.onCreateProjectClicked()

            self.assertEqual(list(destination.iterdir()), [])

    def test_keypoint_names_are_padded_to_the_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            page = self._page(temporary, keypoints=4)
            page.keypoint_manager.input_rows[0]["input_field"].setText("nose")

            config = page.getAllProjectConfig()

            # The annotator reads names positionally, so a short list left keypoints unnamed.
            self.assertEqual(len(config["keypoints_name"]), config["keypoints"])
            self.assertEqual(config["keypoints_name"][0], "nose")


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class PublicDatasetConfigTests(unittest.TestCase):
    """Every dataset the public-project page offers has to be usable."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_every_offered_dataset_config_parses(self):
        from animalposetracker.cfg import DATA_YAML_PATHS

        for name, path in DATA_YAML_PATHS.items():
            with self.subTest(dataset=name):
                # `acinoset.yaml` was missing commas in its skeleton flow sequence. It is also
                # the first entry, so it was the page's default and its Create always failed.
                parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
                self.assertIsInstance(parsed, dict)

    def test_the_default_dataset_selection_parses(self):
        page = PublicDatasetProjectPage()

        from animalposetracker.cfg import DATA_YAML_PATHS

        self.assertIn(page.dataset, DATA_YAML_PATHS)
        yaml.safe_load(Path(DATA_YAML_PATHS[page.dataset]).read_text(encoding="utf-8"))


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class AnnotatorRoundTripTests(unittest.TestCase):
    """What the annotator writes has to be what it can read back."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def _page(self, temporary, keypoints=("nose", "ear"), skeleton=((0, 1),)):
        image = Path(temporary) / "frame1.jpg"
        cv2.imwrite(str(image), np.zeros((40, 60, 3), dtype=np.uint8))
        config = Path(temporary) / "project.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "keypoints_name": list(keypoints),
                    "classes_name": ["mouse"],
                    "skeleton": [list(edge) for edge in skeleton],
                    "keypoints": len(keypoints),
                    "classes": 1,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        page = AnnotatorPage()
        page.config_path = str(config)
        page.project_config = yaml.safe_load(config.read_text(encoding="utf-8"))
        page._populate_comboboxes()
        page.images = {image.name: str(image)}
        page.sorted_keys = sorted(page.images)
        page.current_image_index = 0
        page.current_image = str(image)
        page.current_image_path = str(temporary)
        page.labels_cache = {}
        return page, image, config

    @staticmethod
    def _target(keypoints, box=True):
        target = AnnotationTarget(0, None)
        if box:
            target.bounding_rect = {"category_id": 0, "bbox": [1, 2, 10, 20], "area": 200}
        target.keypoints = {
            index: {"name": name, "pos": [5 + index, 6 + index, 2]}
            for index, name in enumerate(keypoints)
        }
        return target

    def test_a_boxless_target_survives_save_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "animalposetracker.gui.annotatorpage.QMessageBox"
        ):
            page, image, _config = self._page(temporary)
            page.drawing_label.targets = [self._target(("nose", "ear"), box=False)]

            page.onSaveLabel()

            # The file used to record `"bbox": null`, and the loader then raised on `len(None)`
            # and dropped every annotation for the image.
            page._LoadAnnotationtoCache(
                str(image), json.loads(Path(image).with_suffix(".json").read_text(encoding="utf-8"))
            )

            cached = page.labels_cache[str(image)]["targets"]
            self.assertEqual(len(cached), 1)
            self.assertEqual(len(cached[0].keypoints), 2)

    def test_deleting_annotations_persists(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "animalposetracker.gui.annotatorpage.QMessageBox"
        ):
            page, image, _config = self._page(temporary)
            page.drawing_label.targets = [self._target(("nose", "ear"))]
            page.onSaveLabel()

            page.onDeleteLabel()

            written = json.loads(Path(image).with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(written["annotations"], [])

    def test_deleting_a_referenced_keypoint_row_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "animalposetracker.gui.annotatorpage.QMessageBox"
        ) as box:
            page, _image, config = self._page(
                temporary, keypoints=("a", "b", "c"), skeleton=((0, 2),)
            )
            page.EditKeypoints.setText("Save Keypoints")
            page._HandleConfigDisplay("keypoints")

            # Index 1 sits between two referenced keypoints; removing it would shift `c` from 2
            # to 1 while `skeleton` kept saying 2, so the edge would mean a different keypoint.
            page._delete_config_row(page.ConfigureDisplay.topLevelItem(1))

            self.assertTrue(box.warning.called)
            saved = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(saved["keypoints_name"], ["a", "b", "c"])

    def test_deleting_an_unreferenced_keypoint_row_renumbers_consistently(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "animalposetracker.gui.annotatorpage.QMessageBox"
        ):
            page, _image, config = self._page(
                temporary, keypoints=("a", "b", "c"), skeleton=((0, 1),)
            )
            page.EditKeypoints.setText("Save Keypoints")
            page._HandleConfigDisplay("keypoints")

            # Index 2 is past every referenced index, so removing it shifts nothing.
            page._delete_config_row(page.ConfigureDisplay.topLevelItem(2))
            page._SaveConfigChanges("keypoints")

            saved = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(saved["keypoints_name"], ["a", "b"])
            for source, destination in saved["skeleton"]:
                self.assertLess(source, len(saved["keypoints_name"]))
                self.assertLess(destination, len(saved["keypoints_name"]))

    def test_editing_classes_then_keypoints_writes_the_right_key(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "animalposetracker.gui.annotatorpage.QMessageBox"
        ):
            page, _image, config = self._page(temporary)

            # Open the Classes editor, then the Keypoints one, the way the two buttons do.
            with patch.object(page, "sender", return_value=page.EditClasses):
                page.onEditParameters()
            with patch.object(page, "sender", return_value=page.EditKeypoints):
                page.onEditParameters()

            page.ConfigureDisplay.topLevelItem(0).setText(1, "nose")
            with patch.object(page, "sender", return_value=page.EditKeypoints):
                page.onEditParameters()  # save

            saved = yaml.safe_load(config.read_text(encoding="utf-8"))
            # One shared edit flag meant this save wrote the *keypoints* list into whichever key
            # the button pair happened to leave active -- the class list could end up in
            # `keypoints_name` and a skeleton list in `classes_name`.
            self.assertEqual(saved["keypoints_name"], ["nose", "ear"])
            self.assertEqual(saved["classes_name"], ["mouse"])


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class LayoutGuardrailTests(unittest.TestCase):
    """No widget or row may be built and left out of its page."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _left_out(page):
        reachable_layouts, reachable_widgets = set(), set()

        def visit_widget(widget):
            reachable_widgets.add(id(widget))
            # The walk has to descend into containers that hold their pages indirectly, or every
            # control inside a scroll area reads as "left out".
            if isinstance(widget, QScrollArea) and widget.widget() is not None:
                inner = widget.widget()
                reachable_widgets.add(id(inner))
                if inner.layout() is not None:
                    walk(inner.layout())
            if isinstance(widget, (QTabWidget, QStackedWidget)):
                for index in range(widget.count()):
                    visit_widget(widget.widget(index))
            if widget.layout() is not None:
                walk(widget.layout())

        def walk(layout):
            reachable_layouts.add(id(layout))
            for index in range(layout.count()):
                item = layout.itemAt(index)
                widget = item.widget()
                if widget is not None:
                    visit_widget(widget)
                elif item.layout() is not None:
                    walk(item.layout())

        if page.layout() is not None:
            walk(page.layout())

        layouts = [
            name
            for name, value in sorted(vars(page).items())
            if isinstance(value, QLayout) and id(value) not in reachable_layouts
        ]
        widgets = [
            name
            for name, value in sorted(vars(page).items())
            if isinstance(value, QWidget)
            and id(value) not in reachable_widgets
            # Menus and the menu bar are not mounted in a layout by design, and the annotator's
            # drawing board is placed on its graphics scene rather than in a layout.
            and not isinstance(value, (QMenu, QMenuBar))
            and name != "drawing_label"
        ]
        return layouts + widgets

    def test_no_page_leaves_a_widget_or_row_out(self):
        for name, page_class in (
            ("mainpage", AnimalPoseTrackerPage),
            ("inferencer", InferencerPage),
            ("annotator", AnnotatorPage),
            ("createnewproject", CreateNewProjectPage),
            ("publicdatasets", PublicDatasetProjectPage),
        ):
            with self.subTest(page=name):
                self.assertEqual(self._left_out(page_class()), [])


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class ThemeTests(unittest.TestCase):
    """A tool window is top-level, so it has to be given the theme rather than inheriting it."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_a_tool_window_carries_the_main_window_theme(self):
        from animalposetracker.gui import DARK_THEME_PATH

        page = AnimalPoseTrackerPage()
        page.onChangeTheme("dark")

        window = page._open_sub_page(InferencerPage)

        self.assertEqual(window.styleSheet(), Path(DARK_THEME_PATH).read_text(encoding="utf-8"))
        window.close()

    def test_the_chosen_theme_survives_a_restart(self):
        from PySide6.QtCore import QSettings

        settings = QSettings("Northeast Normal University", "AnimalPoseTracker")
        previous = settings.value("theme")
        try:
            page = AnimalPoseTrackerPage()
            page.onChangeTheme("dark")
            self.assertEqual(settings.value("theme"), "dark")

            reopened = AnimalPoseTrackerPage()
            self.assertEqual(
                reopened.styleSheet(),
                Path(__import__(
                    "animalposetracker.gui", fromlist=["DARK_THEME_PATH"]
                ).DARK_THEME_PATH).read_text(encoding="utf-8"),
            )
        finally:
            if previous is None:
                settings.remove("theme")
            else:
                settings.setValue("theme", previous)

    def test_every_page_sets_its_own_window_icon(self):
        # The icon is a file on disk and used to be baked into the generated `ui_*.py`, which
        # meant regenerating them from the Designer sources dropped it.
        for page_class in (
            AnimalPoseTrackerPage,
            InferencerPage,
            AnnotatorPage,
            CreateNewProjectPage,
            PublicDatasetProjectPage,
        ):
            with self.subTest(page=page_class.__name__):
                self.assertFalse(page_class().windowIcon().isNull())


@unittest.skipUnless(QApplication is not None, "PySide6 (the gui extra) is not installed")
class UiModuleTests(unittest.TestCase):
    """The `ui_*.py` modules are the GUI's source of truth.

    The Designer `.ui` sources were retired on 2026-10-15; the modules are maintained
    directly now, so the old "in sync with the generated output" check is gone with them.
    """

    def test_no_ui_module_hard_codes_a_machine_path(self):
        root = Path(__file__).resolve().parent.parent
        for generated in sorted((root / "animalposetracker" / "gui").glob("ui_*.py")):
            text = generated.read_text(encoding="utf-8")
            with self.subTest(file=generated.name):
                # A Designer backup path was once baked in here, so regenerating on any other
                # machine produced an icon that could not load.
                self.assertNotIn(".designer", text)
                self.assertNotIn(":\\", text)


if __name__ == "__main__":
    unittest.main()
