"""Headless checks for the PySide6 pages.

Offscreen Qt keeps these runnable without a display, and the module is skipped outright when
PySide6 is missing -- it ships as the optional ``gui`` extra, so the core suite must stay
green without it.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    from animalposetracker.gui.createnewprojectpage import CreateNewProjectPage
    from animalposetracker.gui.inferencerpage import InferencerPage
    from animalposetracker.gui.mainpage import AnimalPoseTrackerPage
except ImportError:  # pragma: no cover - depends on the optional gui extra
    QApplication = None
    CreateNewProjectPage = None
    InferencerPage = None
    AnimalPoseTrackerPage = None


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
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "project.yaml"
            config_path.write_text("project_name: probe\n", encoding="utf-8")
            weights_path = Path(temporary) / "best.onnx"
            weights_path.write_bytes(b"")

            page = InferencerPage()
            self.assertFalse(page.Start.isEnabled())

            with patch("os.chdir"), patch(
                "animalposetracker.gui.inferencerpage.QFileDialog.getOpenFileName",
                return_value=(str(config_path), ""),
            ):
                page.onSelectConfigureClicked()
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

            self.assertTrue(page.Start.isEnabled())


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


if __name__ == "__main__":
    unittest.main()
