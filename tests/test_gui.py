"""Headless checks for the PySide6 pages and the Qt threads they drive.

Offscreen Qt keeps these runnable without a display, and the module is skipped outright when
PySide6 is missing -- it ships as the optional ``gui`` extra, so the core suite must stay
green without it.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

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
