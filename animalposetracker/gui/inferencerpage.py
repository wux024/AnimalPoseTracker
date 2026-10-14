from PySide6.QtCore import QMetaObject, Qt, Q_ARG, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QPalette
from PySide6.QtWidgets import  (QApplication, QFileDialog, QWidget, QMessageBox,
                                QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                                QLabel, QSpinBox)
from dataclasses import replace
import json
import os
import cv2
import numpy as np
import sys
from pathlib import Path
import queue

from .ui_inferencer import Ui_Inferencer
from animalposetracker.gui import LOGO_SMALL_PATH
from animalposetracker.utils import (VideoReaderThread, VideoWriterThread, 
                                     PreprocessThread, InferenceThread, 
                                     PostprocessThread, VisualizeThread)
from animalposetracker.inference import ENGINEtoDEVICE
from animalposetracker.inference.device_resolution import detect_platform
from animalposetracker.inference.session import (
    ENGINE_NAMES,
    build_inference_session,
    instance_to_record,
    metadata_head,
    pixel_points,
    read_inference_config,
    validate_inference_selection,
)
from animalposetracker.postprocessing.output_decoders import decode_pose_outputs
from animalposetracker.rendering import INFERENCE_STYLE, draw_instance, pixel_box
class InferencerPage(QWidget, Ui_Inferencer):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)

        # init 
        self.initialize_controls()
    
    def initialize_controls(self):
        """Initialize all buttons and dependent controls"""
        os.chdir(Path.cwd())

        # Set here rather than in the Designer file: the icon is a file on disk, and a path baked
        # into `ui_*.py` breaks as soon as the sources are regenerated.
        self.setWindowIcon(QIcon(LOGO_SMALL_PATH))

        # constants
        self.camera_list = {}
        self.supported_engines_and_devices = {}
        self.visualization_config = {}
        self.max_cache_size = 1024

        self.session = None
        self._class_colors = ()
        self._edge_colors = ()
        self._keypoint_colors = ()
        self.device = None
        self.engine = None

        # threads
        self.videoreader_thread = None
        self.videowriter_thread = None
        self.preprocess_thread = None
        self.inference_thread = None
        self.postprocess_thread = None
        self.visualize_thread = None
        self.display_thread = None
        self.write_frame_end = True
        # True between a successful Start and the stop that finalizes it.
        self._run_active = False

        self.data_config_path = None
        self.weights_path = None
        self.detector_path = None
        # The mode switch and its dependents: one-stage by default; the boxes on the right
        # follow the mode, never the other way round.
        self._two_stage = False
        self._decode_params_active = False
        # Whether the chosen pose artifact is a SimCC (AnimalViTPose) one; the stage pair gate
        # in `_refresh_start_state` needs it before any artifact is loaded or built.
        self._weights_is_simcc = False
        # The shared two-stage stream (detector -> crops -> SimCC), built at Start and dropped
        # when the run ends.
        self._stream_pipeline = None
        # Whether each half of the inference setup has passed its checks; Start needs both.
        self._config_ready = False
        self._weights_ready = False

        # The camera controls normally fill these in when the user touches them or when a source
        # is opened. Seeding them from the spin boxes keeps Start working when neither happened:
        # they used to exist only after their own handler had run, so reading them threw.
        self.width = self.WidthSetup.value()
        self.height = self.HeightSetup.value()
        self.fps = self.FPSSetup.value()

        # One run's frame counter and record buffer, both reset by Start.
        self._frame_index = 0
        self.records = []
        self.tracker = None
        # Frozen by `_build_palettes` when a run starts; these defaults keep the render path
        # usable before then.
        self._run_style = None
        self._run_class_names = ()
        self._run_skeleton = ()
        self._add_save_switch()
        self._add_decode_param_controls()
        # The filter's widgets are built first, because the tracker's builder is the one that
        # places them (both rows go into the left group, in this order).
        self._add_filter_controls()
        self._add_tracker_controls()
        self._add_status_line()
        self._align_labels()
        self._align_display_switches()

        self.platform = detect_platform()
        self.device_check_results = self._detect_device_availability()

        self.tools_disabled()
        self.SelectConfigure.setEnabled(True)
        # The two model buttons open up with the config -- it states the dataset and may
        # already declare both weights files.
        self.SelectPoser.setEnabled(False)
        self.SelectDetector.setEnabled(False)
        self.ModeToggle.setText("Mode: One-stage")
        # The live copies the decode stage reads each frame; the boxes' handlers keep them in
        # step from here on.
        self._live_poser_conf = float(self.Conf.value())
        self._live_nms_iou = float(self.IoU.value())
        self._live_detector_conf = float(self.DetectorConf.value())
        self.ShowKeypoints.setCheckState(Qt.CheckState.Checked)
        self.Show.setCheckState(Qt.CheckState.Checked)
        self.FPSShow.setCheckState(Qt.CheckState.Checked)
        self.PreprocessTime.setCheckState(Qt.CheckState.Checked)
        self.InferenceTime.setCheckState(Qt.CheckState.Checked)
        self.PostprocessTime.setCheckState(Qt.CheckState.Checked)
        self.CameraORVideos.setAutoExclusive(False)

        # Connect all signals to their respective slots
        self.setupConnections()

        # iningitialize visualization config
        self._init_visualization_config()

    def setupConnections(self):
        """Connect all UI signals to their corresponding slot functions"""
        
        # Base configuration signals
        self.SelectConfigure.clicked.connect(self.onSelectConfigureClicked)
        self.SelectDetector.clicked.connect(self.onSelectDetectorClicked)
        self.SelectPoser.clicked.connect(self.onSelectPoserClicked)
        self.ModeToggle.clicked.connect(self.onModeToggled)
        self.CameraORVideos.toggled.connect(self.onCameraORVideosToggled)
        self.CameraVideosSelection.currentIndexChanged.connect(self.onCameraVideosSelectionChanged)
        self.CheckCameraVideosConnect.clicked.connect(self.onCheckCameraVideosConnectClicked)
        
        # Camera parameter signals
        self.WidthSetup.valueChanged.connect(self.onWidthSetupChanged)
        self.HeightSetup.valueChanged.connect(self.onHeightSetupChanged)
        self.FPSSetup.valueChanged.connect(self.onFPSSetupChanged)
        
        # Engine and device signals
        self.ModelBits.clear()
        self.EngineSelection.clear()
        self.DeviceSelection.clear()
        self.ModelBits.currentTextChanged.connect(self.onModelBitsChanged)
        self.EngineSelection.currentTextChanged.connect(self.onEngineSelectionChanged)
        self.DeviceSelection.currentTextChanged.connect(self.onDeviceSelectionChanged)
        
        # Start/Stop control signals
        self.Start.clicked.connect(self.onStartClicked)
        self.End.clicked.connect(self.onEndClicked)
        
        # Display option signals
        self.IoU.valueChanged.connect(self.onIoUChanged)
        self.Conf.valueChanged.connect(self.onConfChanged)
        self.DetectorConf.valueChanged.connect(self.onDetectorConfChanged)
        self.Backgroud.stateChanged.connect(self.onBackgroudStateChanged)
        self.ShowClasses.stateChanged.connect(self.onShowClassesStateChanged)
        self.ShowKeypoints.stateChanged.connect(self.onShowKeypointsStateChanged)
        self.ShowKeypointsRadius.valueChanged.connect(self.onShowKeypointsRadiusChanged)
        self.ShowSkeletons.stateChanged.connect(self.onShowSkeletonsStateChanged)
        self.ShowSkeletonsLineWidth.valueChanged.connect(self.onShowSkeletonsLineWidthChanged)
        self.ShowBBox.stateChanged.connect(self.onShowBBoxStateChanged)
        self.ShowBBoxWidth.valueChanged.connect(self.onShowBBoxWidthChanged)
        self.EnableTracking.stateChanged.connect(self.onEnableTrackingStateChanged)
        self.EnableFilter.stateChanged.connect(self.onEnableFilterStateChanged)
        self.FilterMethod.currentTextChanged.connect(self.onFilterMethodChanged)
        self.FPSShow.stateChanged.connect(self.onFPSShowStateChanged)
        self.PreprocessTime.stateChanged.connect(self.onPreprocessTimeStateChanged)
        self.InferenceTime.stateChanged.connect(self.onInferenceTimeStateChanged)
        self.PostprocessTime.stateChanged.connect(self.onPostprocessTimeStateChanged)
        self.FontScale.valueChanged.connect(self.onFontScaleChanged)

    def tools_enabled(self):
        self.CheckCameraVideosConnect.setEnabled(True)
        self.WidthSetup.setEnabled(True)
        self.HeightSetup.setEnabled(True)
        self.FPSSetup.setEnabled(True)
        # Start is deliberately absent: it is lit only by `_refresh_start_state`, once a config
        # *and* weights have both passed their checks. Enabling it here let the user start a run
        # with neither chosen, and the builder then failed on a None path.
        self.End.setEnabled(True)
        self.Save.setEnabled(True)
        # The decode trio follows the mode; the full matrix lives in
        # `_refresh_decode_param_states`.
        self._refresh_decode_param_states(True)
        self.Show.setEnabled(True)
        self.Backgroud.setEnabled(True)
        self.ShowClasses.setEnabled(True)
        self.ShowKeypoints.setEnabled(True)
        self.ShowKeypointsRadius.setEnabled(True)
        self.ShowSkeletons.setEnabled(True)
        self.ShowSkeletonsLineWidth.setEnabled(True)
        self.ShowBBox.setEnabled(True)
        self.ShowBBoxWidth.setEnabled(True)
        self.FPSShow.setEnabled(True)
        self.PreprocessTime.setEnabled(True)
        self.InferenceTime.setEnabled(True)
        self.PostprocessTime.setEnabled(True)
        self.EnableTracking.setEnabled(True)
        self.TrackerAlgorithm.setEnabled(self.EnableTracking.isChecked())
        self._refresh_filter_controls()

    def tools_disabled(self):
        self.CheckCameraVideosConnect.setEnabled(False)
        self.WidthSetup.setEnabled(False)
        self.HeightSetup.setEnabled(False)
        self.FPSSetup.setEnabled(False)
        self.Start.setEnabled(False)
        self.End.setEnabled(False)
        self.Save.setEnabled(False)
        self._refresh_decode_param_states(False)
        self.Show.setEnabled(False)
        self.Backgroud.setEnabled(False)
        self.ShowClasses.setEnabled(False)
        self.ShowKeypoints.setEnabled(False)
        self.ShowKeypointsRadius.setEnabled(False)
        self.ShowSkeletons.setEnabled(False)
        self.ShowSkeletonsLineWidth.setEnabled(False)
        self.ShowBBox.setEnabled(False)
        self.ShowBBoxWidth.setEnabled(False)
        self.FPSShow.setEnabled(False)
        self.PreprocessTime.setEnabled(False)
        self.InferenceTime.setEnabled(False)    
        self.PostprocessTime.setEnabled(False)
        self.EnableTracking.setEnabled(False)
        self.TrackerAlgorithm.setEnabled(False)
        for widget in self._filter_controls():
            widget.setEnabled(False)
    def _add_save_switch(self):
        """Name the Save switch and move it in with the display options.

        The records are written out whether or not anyone asks for them, so there is one switch
        here and not two, and it belongs to the row that lists what the run shows -- on its own it
        would have been a lone checkbox in a row of its own.
        """
        # "Save" alone read as ambiguous while there was also a records switch. The label is set
        # here because ui_inferencer.py is generated.
        self.Save.setText("Save Video")
        self.Save.setToolTip(
            "Write the rendered frames to runs/inference/output.avi. "
            "Per-frame pose records always go to runs/inference/records.json, "
            "in the same shape `infer --records` writes"
        )

        # Save has to leave its old row by hand: a layout keeps its own item list, and Qt will not
        # pull the widget out of it. Both rows live in the same group box, so it needs no reparent.
        self.SaveIoUConfLayout.removeWidget(self.Save)
        self.MainShowLayout.insertWidget(0, self.Save)

    def _add_decode_param_controls(self):
        """Gather the decode thresholds into one group at the top of the display column.

        These are the boxes meant to be turned while a run is going: their values are read
        again every frame (see `_decode`), so an edit lands within a frame or two and never
        needs a restart -- that is why they live on the right, next to the display switches,
        the other things that keep reacting while the stream runs. Their labels say exactly
        what each one bites: the NMS threshold only matters for outputs that still carry NMS,
        the detector's filter only for the two-stage stream, and `_refresh_decode_param_states`
        lights the boxes the current mode actually uses.
        """
        # The two rows move out of the old Save row (Save itself already left for the display
        # row in `_add_save_switch`) and are rebuilt as one trio. They stay inside the display
        # group, so no reparenting is needed -- only a layout moved between group boxes must
        # carry its children across.
        self.SaveIoUConfLayout.removeItem(self.IoULayout)
        self.SaveIoUConfLayout.removeItem(self.ConfLayout)
        self.ShowToolsGroupVLayout.removeItem(self.SaveIoUConfLayout)
        del self.SaveIoUConfLayout

        self.IoULabel.setText("NMS IoU")
        self.ConfLabel.setText("Poser Conf")

        self.DetectorConf = QDoubleSpinBox(self.ShowToolsGroup)
        self.DetectorConf.setObjectName("DetectorConf")
        self.DetectorConf.setMaximum(1.0)
        self.DetectorConf.setSingleStep(0.01)
        self.DetectorConf.setValue(0.5)
        self.DetectorConf.setToolTip("Detector confidence threshold: keep only boxes above it (two-stage only; adjustable while running)")
        self.DetectorConfLabel = QLabel("Detector Conf", self.ShowToolsGroup)

        self.DetectorConfLayout = QHBoxLayout()
        self.DetectorConfLayout.setObjectName("DetectorConfLayout")
        self.DetectorConfLayout.addWidget(self.DetectorConfLabel)
        self.DetectorConfLayout.addWidget(self.DetectorConf)

        self.DecodeParamsLayout = QHBoxLayout()
        self.DecodeParamsLayout.setObjectName("DecodeParamsLayout")
        self.DecodeParamsLayout.addLayout(self.IoULayout)
        self.DecodeParamsLayout.addLayout(self.DetectorConfLayout)
        self.DecodeParamsLayout.addLayout(self.ConfLayout)
        self.ShowToolsGroupVLayout.insertLayout(0, self.DecodeParamsLayout)

    def _add_filter_controls(self):
        """Build the temporal filter's controls; the tracker's builder places them.

        Every switch in this group is a checkbox, so the filter gets one of its own: it is the
        on/off, which is why the method list has no `none` -- one way to be off is enough. The
        values and defaults match `infer --pose-filter` and its `--filter-*` arguments, so a look
        tuned here can be written down as a command line and the other way round.

        The smoothing acts on a whole sequence and therefore runs when the run ends; it changes
        the records file, never the live picture.
        """
        self.EnableFilter = QCheckBox(self.BaseToolsGroup)
        self.EnableFilter.setObjectName("EnableFilter")
        self.EnableFilter.setText("Filter")
        self.EnableFilter.setToolTip("Smooth the whole trajectory once when the run ends: only the saved records change, not the live picture")

        self.FilterMethod = QComboBox(self.BaseToolsGroup)
        self.FilterMethod.setObjectName("FilterMethod")
        self.FilterMethod.setMinimumSize(QSize(0, 25))
        self.FilterMethod.addItems(["median", "savgol"])
        self.FilterMethod.setToolTip("median = median smoothing; savgol = Savitzky-Golay fit")

        self.FilterValuesLayout = QHBoxLayout()
        self.FilterValuesLayout.setObjectName("FilterValuesLayout")

        def spin(widget, name, prefix, value, maximum, tooltip):
            widget.setObjectName(name)
            widget.setPrefix(f"{prefix} ")
            widget.setRange(widget.minimum(), maximum)
            widget.setValue(value)
            widget.setToolTip(tooltip)
            # Height only: the width is the row's to divide, as it is for every control the
            # Designer file drew in this group.
            widget.setMinimumSize(QSize(0, 25))
            self.FilterValuesLayout.addWidget(widget)
            return widget

        self.FilterWindow = spin(
            QSpinBox(self.BaseToolsGroup), "FilterWindow", "win", 5, 999, "Smoothing window in frames"
        )
        self.FilterPolyorder = spin(
            QSpinBox(self.BaseToolsGroup), "FilterPolyorder", "poly", 2, 99,
            "Savitzky-Golay polynomial order (savgol only)",
        )
        # Prefixed `kpt`, not `conf`: the decode threshold in the row above is also a confidence.
        self.FilterConfidence = spin(
            QDoubleSpinBox(self.BaseToolsGroup), "FilterConfidence", "kpt", 0.25, 1.0,
            "Keypoints below this confidence are left out of smoothing (not the decode Conf)",
        )
        self.FilterConfidence.setSingleStep(0.05)
        self.FilterConfidence.setDecimals(2)
        self.FilterMaxGap = spin(
            QSpinBox(self.BaseToolsGroup), "FilterMaxGap", "gap", 0, 999,
            "Largest missing-frame gap to interpolate (0 = no interpolation)",
        )

        for widget in self._filter_controls():
            widget.setEnabled(False)

    def _add_tracker_controls(self):
        """Add the online-tracking switch and its algorithm picker.

        The list comes from the same tuple the `--tracker` command line validates against,
        imported here rather than at module scope: the tracking package costs about a second to
        import, and the page would pay it every time the window opens.
        """
        from animalposetracker.tracking.config import ALGORITHMS

        self.EnableTracking = QCheckBox(self.BaseToolsGroup)
        self.EnableTracking.setObjectName("EnableTracking")
        self.EnableTracking.setText("Track")
        self.EnableTracking.setToolTip(
            "Track targets frame by frame: canvas labels gain a track id, and the saved records carry a track_id"
        )
        self.TrackerAlgorithm = QComboBox(self.BaseToolsGroup)
        self.TrackerAlgorithm.setObjectName("TrackerAlgorithm")
        self.TrackerAlgorithm.setMinimumSize(QSize(0, 25))
        self.TrackerAlgorithm.addItems(list(ALGORITHMS))
        self.TrackerAlgorithm.setToolTip("Tracking algorithm; the same list the command line's `infer --tracker` accepts")
        self.TrackerAlgorithm.setEnabled(False)

        # The Designer file leaves an empty TrackerLayout in the left group's engine/device block:
        # a slot kept for a tracker row that no longer belongs there. The slot is reused for the
        # row rather than replaced -- it already carries the right name, and a layout that is in no
        # layout at all draws nothing while reading like an oversight. Qt will not insert a layout
        # that still has a parent, so it comes out of the engine/device block first.
        self.EngineDeviceLayout.removeItem(self.TrackerLayout)

        # One row carries both switches and both pickers: tracking and filtering are the two
        # optional stages a run can have, and the filter has nothing to work on without the
        # tracker, so they read as a pair. The filter's values follow on the row below.
        for widget in (
            self.EnableTracking,
            self.TrackerAlgorithm,
            self.EnableFilter,
            self.FilterMethod,
        ):
            self.TrackerLayout.addWidget(widget)
        self.BaseToolsLayout.insertLayout(2, self.TrackerLayout)
        self.BaseToolsLayout.insertLayout(3, self.FilterValuesLayout)

    def _filter_controls(self):
        return (
            self.EnableFilter,
            self.FilterMethod,
            self.FilterWindow,
            self.FilterPolyorder,
            self.FilterConfidence,
            self.FilterMaxGap,
        )

    def _filter_is_on(self):
        """Whether the filter should run: switched on, and given track ids to work on.

        The workflow refuses records without track ids outright, which is the same rule the
        command line enforces by requiring `--tracker` alongside `--pose-filter`.
        """
        return self.EnableFilter.isChecked() and self.EnableTracking.isChecked()

    def _refresh_filter_controls(self):
        """Follow tracking, follow the filter's own switch, and let the order follow the method."""
        tracking = self.EnableTracking.isChecked() and self.EnableTracking.isEnabled()
        self.EnableFilter.setEnabled(tracking)
        active = tracking and self.EnableFilter.isChecked()
        method = self.FilterMethod.currentText()
        self.FilterMethod.setEnabled(active)
        self.FilterWindow.setEnabled(active)
        self.FilterConfidence.setEnabled(active)
        self.FilterMaxGap.setEnabled(active)
        self.FilterPolyorder.setEnabled(active and method == "savgol")

    def _final_records(self):
        """The records to write: smoothed when the filter is on, exactly as `infer --records`."""
        if not self._filter_is_on() or not self.records:
            return self.records
        from animalposetracker.postprocessing.workflows import filter_tracked_video_records

        return filter_tracked_video_records(
            self.records,
            method=self.FilterMethod.currentText(),
            window_length=int(self.FilterWindow.value()),
            polyorder=int(self.FilterPolyorder.value()),
            confidence_threshold=float(self.FilterConfidence.value()),
            max_gap=int(self.FilterMaxGap.value()),
        )

    def _build_tracker(self):
        """Create this run's tracker, or None when tracking is off.

        Built through the same entry point the CLI uses, and reset per run so a new session
        starts its track IDs from scratch instead of continuing the previous one's.
        """
        from animalposetracker.tracking import create_tracker

        if not self.EnableTracking.isChecked():
            self.tracker = None
            return
        self.tracker = create_tracker(self.TrackerAlgorithm.currentText())
        self.tracker.reset()

    def _add_status_line(self):
        """One visible line for worker messages.

        `thread_status` only printed, and a packaged GUI has no console to print to: a camera
        that failed to open, a worker error or a written records file was invisible.
        """
        self.StatusLabel = QLabel()
        self.StatusLabel.setObjectName("StatusLabel")
        self.StatusLabel.setWordWrap(True)
        palette = self.StatusLabel.palette()
        # Greyed through the palette so it follows the active theme instead of a fixed colour.
        palette.setColor(
            QPalette.WindowText, palette.color(QPalette.Disabled, QPalette.WindowText)
        )
        self.StatusLabel.setPalette(palette)
        self.InferencerLayout.addWidget(self.StatusLabel)

    def _align_labels(self):
        """Put the left group's labels in one column, wide enough for the longest of them.

        Rows read as a table when the labels stand in a column and the fields all start at the same
        x. The Designer file does that for the display group (its label column is a fixed 120). The
        left group's labels were left to their own text instead, so the field column began at 60 in
        the threshold row, 74 in the camera row and 101 in the engine row.
        """
        labels = (
            self.WidthLabel,
            self.HeightLabel,
            self.FPSLabel,
            self.ModelBitsLabel,
            self.EngineLabel,
            self.DeviceLabel,
        )
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setFixedWidth(width)

        # The decode trio now lives in the display column instead; its three labels align
        # among themselves, to the widest of them.
        trio = (self.IoULabel, self.DetectorConfLabel, self.ConfLabel)
        trio_width = max(label.sizeHint().width() for label in trio)
        for label in trio:
            label.setFixedWidth(trio_width)

    def _align_display_switches(self):
        """Give the display group's two switch rows one column rhythm.

        Both rows open with a fixed 120 cell -- Save keeps the width the Designer gave it, and
        the Font Scale row below starts its slider at 120 as well -- so the FPS switch, the other
        row's first cell, gets the same 120. Each row then divides what is left between three
        equal cells, which lines the two rows up column for column.
        """
        self.FPSShow.setFixedWidth(120)

    def onEnableTrackingStateChanged(self, state):
        """The algorithm picker only means anything while tracking is on."""
        self.TrackerAlgorithm.setEnabled(bool(state) and self.EnableTracking.isEnabled())
        self._refresh_filter_controls()

    def onFilterMethodChanged(self, method):
        """The polynomial order is a Savitzky-Golay setting."""
        self._refresh_filter_controls()

    def onEnableFilterStateChanged(self, state):
        """The filter's own switch decides whether its values matter."""
        self._refresh_filter_controls()

    def _write_records(self):
        """Write this run's per-frame records, where the `infer` command line would put them.

        The file carries the smoothed records when the filter ran, which is what `infer
        --records --pose-filter` writes too: the whole point of the file is to be the run's
        result, and the result is what the filter produced. There is no switch for this -- the
        records are the run's result, so it is written whenever a run produced any.
        """
        if not self.records:
            return
        smoothing = self._filter_is_on()
        try:
            records = self._final_records()
        except (ValueError, KeyError) as error:
            # Records without track ids are refused by the workflow. The controls already prevent
            # that pairing, so this is the belt to their braces -- and it must not cost a whole
            # run's records on the way out.
            QMessageBox.warning(self, "Smoothing not applied", str(error))
            records = self.records

        output_dir = Path.cwd() / "runs" / "inference"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "records.json"
            target.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as error:
            QMessageBox.warning(self, "Saving records failed", f"Could not write the per-frame records:\n{error}")
            return
        note = " (smoothed)" if smoothing else ""
        self.thread_status(f"Per-frame records written to {target}{note} ({len(records)} frames)")

    def onSelectConfigureClicked(self):
        """
        Open file dialog to select YAML configuration file for data settings.
        
        Features:
        - Supports .yaml and .yml extensions
        - Validates YAML structure
        - Stores path and updates UI status
        - Enables dependent controls when successful
        """
        # Open file dialog with YAML filter
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Configuration File",
            dir=os.getcwd(),  # Start in current working directory
            filter="YAML Files (*.yaml *.yml);;All Files (*)",
            options=QFileDialog.Options()
        )
        
        if not file_path:  # User cancelled
            return
        
        # The working directory is left alone: this used to `chdir` into the config's folder, so
        # every `Path.cwd()`-based output (the records and the video) landed next to whichever
        # file was picked last instead of in the project.
        # Store configuration
        self.data_config_path = file_path
  
        # Update UI: both model buttons open with the config, and any weights the config
        # declares are adopted straight away -- a later pick still overrides them.
        self.SelectPoser.setEnabled(True)
        self.SelectDetector.setEnabled(True)
        self._apply_config_declared_paths()
        self._refresh_start_state(show_dialog=True)

    def _refresh_start_state(self, show_dialog=False):
        """Check the chosen config/weights pair and light Start only when both pass.

        Every path that changes either selection comes through here. The two scattered
        `Start.setEnabled(True)` calls this replaces could light the button for a pair that then
        failed at Start -- which is the failure this check exists to move earlier, to where the
        user can still see which file caused it.

        Errors are reported in a dialog; warnings only reach the console, because this page has
        no status bar to put them in and a common one (an artifact with no sidecar) would
        otherwise greet every single selection with a popup.
        """
        report = validate_inference_selection(
            self.data_config_path, self.weights_path, self.detector_path
        )
        self._config_ready = report.config_ok
        self._weights_ready = report.weights_ok
        if self._config_ready and self._weights_ready:
            # These options depend on the model being chosen, not on a source existing, so they
            # become usable as soon as both halves are in place -- which is what makes the rows
            # sitting above the source picker read in the right order.
            self.tools_enabled()
        # The stage pair has to match: SimCC weights run only in the two-stage stream, which
        # additionally needs the detector's weights, while the one-stage stream refuses SimCC.
        # Start stays dark rather than quietly running something other than what the switch
        # says -- and the status line names whichever half the mode is still missing, so the
        # dark button is not a dead end.
        if self._two_stage:
            stage_ok = self._weights_is_simcc and bool(self.detector_path)
        else:
            stage_ok = not self._weights_is_simcc
        self.Start.setEnabled(self._config_ready and self._weights_ready and stage_ok)
        if not stage_ok:
            if self._two_stage and not self._weights_is_simcc:
                self.thread_status(
                    "Two-stage needs SimCC (AnimalViTPose) pose weights: pick different pose weights or switch back to One-stage"
                )
            elif self._two_stage:
                self.thread_status("Two-stage mode: select a detector (Select Detector)")
            else:
                self.thread_status(
                    "SimCC (AnimalViTPose) weights need two-stage: switch Mode to Two-stage"
                )

        for warning in report.warnings:
            self.thread_status(f"Warning: {warning}")
        if report.errors and show_dialog:
            detail = "\n".join(report.errors)
            if report.warnings:
                detail += "\n\nAlso note:\n" + "\n".join(report.warnings)
            QMessageBox.critical(self, "Configuration check failed", detail)
            

    def onSelectPoserClicked(self):
        """Open file dialog to select model weights and 
        auto-configure supported engines and devices.

        Supported Engines:
        - OpenVINO (.onnx and .xml)
        - OpenCV (.onnx)
        - CANN (.om)
        - ONNX (.onnx)
        - TensorRT (.engine)
        - CoreML (.mlmodel and .mlpackage)

        The function will:
        1. Detect file format and hardware availability
        2. Filter available engines and devices based on format and platform
        3. Set default engine based on optimal performance
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Weights",
            "",
            "All Supported Formats (*.xml *.onnx *.om *.engine *.mlmodel *.mlpackage);;"
            "ONNX (*.onnx);;"
            "CANN Model (*.om);;"
            "TensorRT (*.engine *.onnx);;"
            "OpenVINO (*.xml);;"
            "CoreML (*.mlmodel *.mlpackage);;"
            "All Files (*)",
            options=options
        )

        if not file_path:
            return

        self._apply_poser_selection(file_path)

    def _apply_poser_selection(self, file_path):
        """Adopt a pose-model file: validate it, probe the engines, light what depends on it.

        The dialog and the config loader both land here, so a declared weights file gets the
        same checks and side effects as a hand-picked one.
        """
        self.weights_path = file_path
        # Check before probing the hardware: a pair that cannot run should say so now, and the
        # probe is the expensive part of this slot. The detector is part of the check -- it is
        # what turns a SimCC artifact from a dead end into a usable pair.
        report = validate_inference_selection(
            self.data_config_path, self.weights_path, self.detector_path
        )
        self._weights_is_simcc = metadata_head(report.metadata) == "SimCCHead"
        if not report.weights_ok:
            self._config_ready = report.config_ok
            self._weights_ready = False
            self.Start.setEnabled(False)
            self._clear_selections()
            QMessageBox.critical(self, "Configuration check failed", "\n".join(report.errors))
            return

        self._enable_widgets()
        self._clear_selections()
        self.supported_engines_and_devices = {}

        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        

        self._get_supported_engines_and_devices(file_ext)

        if not self.supported_engines_and_devices:
            # Reached when the artifact format is known but no engine on this machine can run it
            # (a TensorRT engine on a box with no NVIDIA GPU, for instance). Reported instead of
            # raising, which only printed a traceback.
            QMessageBox.warning(
                self, "No usable engine",
                f"{file_ext} artifacts have no usable inference engine on this machine; "
                "choose a format whose runtime is installed (for example ONNX)."
            )
            self._clear_selections()
            self._refresh_start_state(show_dialog=False)
            return

        self._init_engine_selection(list(self.supported_engines_and_devices.keys()))
        self._init_available_devices()
        self._init_model_bits()

        # Start needs both halves of the configuration. This used to test an attribute that is
        # never assigned anywhere, so it stayed disabled until a camera or video was touched.
        self._refresh_start_state(show_dialog=False)

    def onSelectDetectorClicked(self):
        """Pick the detector artifact that turns the run into the two-stage stream."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Detector Weights",
            "",
            "ONNX (*.onnx);;All Files (*)",
            options=QFileDialog.Options(),
        )
        if not file_path:
            return
        self._apply_detector_selection(file_path)

    def _apply_detector_selection(self, file_path):
        """Adopt a detector file; no detector at all is what keeps the run one-stage."""
        detector = Path(file_path)
        if not detector.is_file():
            QMessageBox.warning(self, "Invalid detector", f"Detector file not found:\n{detector}")
            return
        self.detector_path = str(detector)
        self.thread_status(f"Detector selected: {detector.name}")
        # A pose model refused a moment ago only because the detector was missing deserves its
        # full pass now -- otherwise Start could light with the engine combos never populated.
        if self.weights_path and not self._weights_ready:
            self._apply_poser_selection(self.weights_path)
        self._refresh_start_state(show_dialog=False)

    def _apply_config_declared_paths(self):
        """Adopt the weights the config declares, if any; a later pick still overrides them.

        `read_inference_config` resolves relative paths against the config file's directory,
        so a yaml that points at its neighbours keeps working after the pair is moved.
        """
        try:
            summary = read_inference_config(self.data_config_path)
        except Exception:
            # A broken config is reported by `_refresh_start_state` right after this.
            return
        # The detector first: a declared SimCC poser is only adoptable once the detector is
        # on hand, and the other order would pop a refusal for a pair that is in fact complete.
        if summary.detector_weights_path:
            self._apply_detector_selection(summary.detector_weights_path)
        if summary.weights_path:
            self._apply_poser_selection(summary.weights_path)

    @staticmethod
    def _detector_conventions(detector_path):
        """The declarations an ONNX detector is described by, picked by its file name.

        The NMS-free YOLO exports (``nms=False``) speak pixels + letterbox + xyxy; the RT-DETR
        family speaks normalized + stretch + cxcywh, which is also the fallback when the name
        says nothing. Explicit declarations in the config's ``detector:`` section will sit
        above these presets once that wiring lands; the file name is what there is today.
        """
        name = Path(detector_path).name.lower()
        if "yolo" in name:
            return {
                "box_format": "xyxy",
                "coord_mode": "pixels",
                "preprocess_mode": "letterbox",
            }
        return {
            "box_format": "cxcywh",
            "coord_mode": "normalized",
            "preprocess_mode": "stretch",
        }

    def onModeToggled(self):
        """Flip the stream between one stage and two.

        The mode is the operator's explicit call: it is never inferred from which files are
        set, and loading a config does not move this switch. Two-stage makes the detector's
        weights a requirement; in one-stage a chosen detector simply waits its turn.
        """
        self._two_stage = not self._two_stage
        self.ModeToggle.setText("Mode: Two-stage" if self._two_stage else "Mode: One-stage")
        self._refresh_decode_param_states()
        # The stage-pair explanation lives in `_refresh_start_state` alone -- every path that
        # can leave Start dark goes through it, and it names the missing half.
        self._refresh_start_state(show_dialog=False)

    def _refresh_decode_param_states(self, active=None):
        """Light only the decode boxes the current mode uses (lit only when they have an effect)."""
        if active is None:
            active = self._decode_params_active
        self._decode_params_active = bool(active)
        one_stage = not self._two_stage
        self.IoU.setEnabled(self._decode_params_active and one_stage)
        self.Conf.setEnabled(self._decode_params_active and one_stage)
        self.DetectorConf.setEnabled(self._decode_params_active and not one_stage)

    def onDetectorConfChanged(self, value):
        """Keep the detector's live threshold in step with its box."""
        self._live_detector_conf = float(value)

    def _enable_widgets(self):
        self.EngineSelection.setEnabled(True)
        self.DeviceSelection.setEnabled(True)
        self.CameraORVideos.setEnabled(True)
        self.CheckCameraVideosConnect.setEnabled(True)

    def _clear_selections(self):
        self.EngineSelection.clear()
        self.DeviceSelection.clear()
    
    def _get_supported_engines_and_devices(self, file_ext):
        """Get supported engines based on file extension.

        The labels come from `ENGINE_NAMES`, which is the shared layer's own vocabulary, so the
        combo and the builder cannot drift apart.
        """
        extensions = {
            '.onnx': ("onnx", "openvino", "tensorrt", "opencv"),
            '.xml': ("openvino",),
            '.om': ("cann", "opencv"),
            '.engine': ("tensorrt",),
            '.mlmodel': ("coreml",),
            '.mlpackage': ("coreml",),
        }
        keys = extensions.get(file_ext)
        if keys is None:
            # A format nothing can run is a message for the user, not an exception out of a slot:
            # PySide6 only prints those, so the window simply did nothing.
            QMessageBox.warning(
                self, "Unsupported format",
                f"Unsupported file format: {file_ext or '(no extension)'}\n"
                "Choose an ONNX / OpenVINO / TensorRT / CANN / CoreML artifact."
            )
            return

        supported_engines = [ENGINE_NAMES[key] for key in keys]

        for engine in supported_engines:
            device_list = self._get_supported_devices(engine)
            if device_list:
                self.supported_engines_and_devices[engine] = device_list
                
    def _init_engine_selection(self, engines):
        self.EngineSelection.addItems(engines)
        default_engine = engines[0]
        self.EngineSelection.setCurrentText(default_engine)

    def _detect_device_availability(self):
        """Check if devices are available for each engine and platform"""
        device_check_results = {
            "Intel GPU": self._check_intel_gpu() if self.platform == 'Intel' else False,
            "Intel NPU": self._check_intel_npu() if self.platform == 'Intel' else False,
            "NVIDIA GPU": False if sys.platform == 'darwin' else self._check_nvidia_gpu(),
            "NVIDIA GPU TensorRT": False if sys.platform == 'darwin' else self._check_nvidia_gpu_tensorrt(),
            "AMD GPU": self._check_amd_gpu() if self.platform == 'AMD' else False,
            "Ascend NPU": False if sys.platform in ['darwin', 'win32'] else self._check_ascend_npu(),
            "Metal": True if sys.platform == 'darwin' else False,
        }
        return device_check_results
    
    def _init_available_devices(self):
        """Update device selection based on chosen engine"""
        current_engine = self.EngineSelection.currentText()
        self.DeviceSelection.clear()
        available_devices = self.supported_engines_and_devices.get(current_engine)
        self.DeviceSelection.addItems(available_devices)
        self.DeviceSelection.setCurrentText(available_devices[0])
    
    def _init_model_bits(self):
        """Initialize model bit selection based on model file size"""
        self.ModelBits.clear()
        self.ModelBits.addItems(['FP32', 'FP16', 'INT8'])
        self.ModelBits.setCurrentText('FP32')
        
    def _get_supported_devices(self, engine):
        supported_devices = []
        if engine in ENGINEtoDEVICE:
            for device in ENGINEtoDEVICE[engine][self.platform]:
                if device in ["Intel CPU", "AMD CPU", "ARM CPU"]:
                    supported_devices.append(device)
                elif self.device_check_results.get(device, False):
                    supported_devices.append(device)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        return supported_devices

    # How long any single hardware probe may take. These run inside `initialize_controls`, on the
    # GUI thread, and a wedged driver made `nvidia-smi` hang the whole window with no timeout.
    _PROBE_TIMEOUT_SECONDS = 5

    def _run_probe(self, cmd, shell=False):
        """Run one hardware probe command and return its stdout, or None if it did not answer."""
        import subprocess

        try:
            result = subprocess.run(
                cmd,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self._PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout or ""

    def _windows_video_controllers(self):
        """The names of this machine's display adapters, via CIM.

        `wmic` was the old way to ask; it is absent from current Windows, so every Intel/AMD GPU
        check silently answered "no" and the device list came up short. The answer is cached
        because the Intel and AMD checks both ask, and each query costs a PowerShell start.
        """
        cached = getattr(self, "_video_controllers_cache", None)
        if cached is not None:
            return cached
        output = self._run_probe(
            [
                'powershell', '-NoProfile', '-NonInteractive', '-Command',
                'Get-CimInstance Win32_VideoController | '
                'Select-Object -ExpandProperty Name',
            ]
        )
        self._video_controllers_cache = output or ""
        return self._video_controllers_cache

    def _check_intel_gpu(self):
        """Check if Intel GPU is available in the system"""
        if os.name == 'nt':
            return "Intel" in self._windows_video_controllers()
        return self._run_probe("lspci | grep -i 'VGA.*Intel'", shell=True) is not None
        
    def _check_intel_npu(self):
        """Check if Intel NPU is available through OpenVINO"""
        if os.name == 'nt':
            # Filtered server-side: enumerating every PnP entity costs ~1.7s, and this runs while
            # the page is being constructed.
            output = self._run_probe(
                [
                    'powershell', '-NoProfile', '-NonInteractive', '-Command',
                    'Get-CimInstance Win32_PnPEntity -Filter "Name LIKE \'%AI Boost%\'" | '
                    'Select-Object -ExpandProperty Name',
                ]
            )
            return "AI Boost" in (output or "")
        return os.path.exists("/dev/accel/accel0") or os.path.exists("/sys/class/accel/accel0")
    
    def _check_amd_gpu(self):
        """Check if AMD GPU is available in the system (Windows/Linux)"""
        if os.name == 'nt':
            output = self._windows_video_controllers()
            return "AMD" in output or "Radeon" in output
        output = self._run_probe("lspci | grep -i 'VGA.*AMD'", shell=True)
        return bool(output)

    def _check_nvidia_gpu(self):
        """Check if NVIDIA GPU is available in the system"""
        # The exit status is the answer. Returning True whenever the command merely existed
        # offered CUDA/TensorRT devices on machines whose driver could not load them.
        return self._run_probe(['nvidia-smi']) is not None
    
    def _check_nvidia_gpu_tensorrt(self):
        """Check if NVIDIA GPU with TensorRT is available in the system"""
        if not self._check_nvidia_gpu():
            return False
        try:
            import tensorrt as trt
        except ImportError:
            return False
        return trt.__version__ is not None

    def _check_ascend_npu(self):
        """Check if CANN environment is available for Ascend NPU"""
        return self._run_probe(['npu-smi', 'info']) is not None

    def onCameraORVideosToggled(self, checked):
        """Handle toggling between camera and video source options"""
        if checked:  # Camera mode
            self._setup_camera_ui()
        else:  # File mode
            self._setup_video_ui()

    def _set_camera_setup(self, cap=None):
        # Set default camera parameters (from design)
        if cap is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            width = 640
            height = 480
            fps = 30
        self.WidthSetup.setRange(1, 9999)
        self.HeightSetup.setRange(1, 9999)
        self.FPSSetup.setRange(1, 500)
        self.WidthSetup.setValue(width if width > 0 else 640)
        self.HeightSetup.setValue(height if height > 0 else 480)
        self.FPSSetup.setValue(fps if fps > 0 else 30)

    def _setup_camera_ui(self):
        """Initialize camera mode UI according to the original design"""
        # Set camera mode toggle state
        self.CameraORVideos.setText("Use Camera")
        self.CheckCameraVideosConnect.setText("Preview Camera")
        
        # Configure camera selection dropdown
        self.CameraVideosSelection.clear()
        self._detect_available_cameras()
        if not self.camera_list:
            QMessageBox.warning(self, "Camera", "No camera detected")
        else:
            self.CameraVideosSelection.addItem("Select camera...")
            self.CameraVideosSelection.addItems(self.camera_list.keys())
            self.CameraVideosSelection.setEnabled(True)
        
        # Set default camera parameters (from design)
        self._set_camera_setup()

    def _setup_video_ui(self):
        """Initialize file mode UI according to the original design"""
        # Set file mode toggle state
        self.CameraORVideos.setText("Use Video")
        
        # Configure file selection
        self.CameraVideosSelection.clear()
        self.CameraVideosSelection.addItem("...")
        self.CameraVideosSelection.addItem("Select video file...")
        self.CameraVideosSelection.setEnabled(True)
        self.CheckCameraVideosConnect.setText("Preview Video")

    def _detect_available_cameras(self):
        """Detects available cameras.   
        Returns:
            list: List of available cameras with their indices as keys.
        """
        system_name = sys.platform
        if system_name == 'win32':
            try:
                from PyCameraList.camera_device import list_video_devices
                self.camera_list ={
                    device[1]: device[0] for device in list_video_devices()
                }
            except ImportError:
                raise ImportError("Please install 'PyCameraList' module")
        elif system_name == 'linux':
            import subprocess
            output = subprocess.check_output("ls /dev/video* 2>/dev/null", 
                                             shell=True, 
                                             text=True)
            video_devices = output.strip().split('\n')
            for device in video_devices:
                try:
                    index = int(device.split('/')[-1].replace('video', ''))
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        name = f"Camera {index}"
                        self.camera_list.update({
                            name: index
                        })
                        cap.release()
                except:
                    continue
        elif system_name == 'darwin':
            index = 0
            while True:
                try:
                    cap = cv2.VideoCapture(index)
                    if not cap.isOpened():
                        break
                    name = f"Camera {index}"
                    self.camera_list.update({
                        name: index
                    })
                    cap.release()
                    index += 1
                except:
                    continue
        else:
            raise ValueError(f"Unsupported platform: {system_name}")
            
    def _select_video_file(self):
        """Handle media file selection"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Media Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            options=options
        )
        
        if not file_path:
            return
        
        self.tools_enabled()
        
        # Store the file path for later use
        self.current_video_path = file_path

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Video", "Failed to open video file")
            return
        
        # Update UI with actual video parameters
        self._set_camera_setup(cap)
        cap.release()

    def _update_camera_params(self):
        """Read and display parameters from selected camera"""
        try:
            current_text = self.CameraVideosSelection.currentText()
            cam_id = self.camera_list[current_text]
            cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                QMessageBox.warning(self, "Camera", "Failed to open camera")
                return
            
            self.tools_enabled()
            self._set_camera_setup(cap)
            cap.release()
            
        except Exception as exc:
            # The fallback used to be silent, with the reason only in a console a packaged GUI
            # does not have: the boxes going back to the defaults looked like a normal result.
            self.thread_status(f"Could not read the camera parameters ({exc}); falling back to the default 640x480 / 30 FPS")
            self._set_camera_setup()
        
    def onCameraVideosSelectionChanged(self):
        """Handle changes in camera/video source selection"""
        if self.CameraVideosSelection.currentText() in self.camera_list.keys():
            self._update_camera_params()
        elif "Select video file..." in self.CameraVideosSelection.currentText():
            self._select_video_file()
            self.CameraVideosSelection.setCurrentText("...")
        else:
            pass
    
    def onCheckCameraVideosConnectClicked(self):
        """Handle when checking camera connection is requested"""

        btn = self.CheckCameraVideosConnect
        current_text = btn.text()
        btn.setEnabled(False)
        try:
            if current_text in {"Preview Camera", "Preview Video"}:
                self._start_preview()
            elif current_text in {"Close Camera", "Close Video"}:
                self._stop_preview()
        finally:
            btn.setEnabled(True)
    
    def _start_preview(self):
        """Start camera/video preview"""
        source = self._get_source()
        if source is None:
            return
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            QMessageBox.warning(self, "Preview", "Failed to open camera/video source")
            return
        if self.CameraORVideos.isChecked():
            self._set_camera_params(cap)
        self.preview_thread = VideoReaderThread()
        self.preview_thread.cap = cap
        self.preview_thread.data_ready.connect(self.display_preview_frame)
        self.preview_thread.finished.connect(self._stop_preview)
        self.preview_thread.start()
        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Close {mode}")

    def _stop_preview(self):
        """Stop camera/video preview"""
        # Guarded: the attribute only exists once a preview has been started, and closing the
        # window without one used to raise.
        if getattr(self, "preview_thread", None) is not None:
            self.preview_thread.safe_stop()
            self.preview_thread = None
        self.Display.clear()
        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Preview {mode}")

    def display_preview_frame(self, frame):
        """Handle new frame from camera/video source"""
        # A frame delivered by the reader thread; the conversion and the label update happen on
        # the GUI thread, which is where the queued call lands.
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = RGB.shape
        QImg = QImage(RGB.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg).scaled(
                self.Display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        QMetaObject.invokeMethod(
                self.Display,
                "setPixmap",
                Qt.QueuedConnection,
                Q_ARG(QPixmap, pixmap)
            )
    

    def _init_visualization_config(self):
        """Initialize visualization configuration"""
        self.visualization_config = {
            'conf': self.Conf.value(),
            'iou': self.IoU.value(),
           'show_classes': self.ShowClasses.isChecked(),
           'show_keypoints': self.ShowKeypoints.isChecked(),
           'show_skeletons': self.ShowSkeletons.isChecked(),
           'show_bbox': self.ShowBBox.isChecked(),
            'radius': self.ShowKeypointsRadius.value(),
           'skeleton_line_width': self.ShowSkeletonsLineWidth.value(),
            'bbox_line_width': self.ShowBBoxWidth.value(),
            'background': self.Backgroud.text(),
            'show_fps': self.FPSShow.isChecked(),
           'show_preprocess_time': self.PreprocessTime.isChecked(),
           'show_inference_time': self.InferenceTime.isChecked(),
           'show_postprocess_time': self.PostprocessTime.isChecked(),
            'font_scale': self.FontScale.value() / 10.0,
        }

    # ------------------------------------------------------------ decode and render adapters
    #
    # The page keeps the frame loop; decoding and drawing come from the shared layers. Decoding
    # goes through the same `decode_pose_outputs` contract the `infer` command line uses, and
    # drawing through `rendering.draw_instance`, so the window shows what the written video and
    # the records say. Only the frame-level extras (background, timing text) live here: they
    # need numpy and are not per instance.

    def _build_palettes(self):
        """One colour per class, per skeleton edge and per keypoint, drawn once per run.

        Also freezes the inputs that cannot change while a run is live -- the render style, the
        class names and the skeleton -- so the hot path does not rebuild them every frame.
        """
        def palette(count):
            if count <= 0:
                return ()
            colours = np.random.uniform(0, 255, size=(count, 3))
            return tuple(tuple(int(channel) for channel in colour) for colour in colours)

        self._class_colors = palette(len(self.session.class_names))
        self._edge_colors = palette(len(self.session.skeleton))
        self._keypoint_colors = palette(self.session.kpt_shape[0])

        self._run_class_names = tuple(self.session.class_names)
        self._run_skeleton = tuple(self.session.skeleton)
        self._run_style = self._render_style(None)

    def _render_style(self, frame=None):
        """The page's look, as data, for the neutral renderer."""
        config = self.visualization_config
        input_height = self.session.input_config.input_size[1]
        return replace(
            INFERENCE_STYLE,
            keypoint_colors=self._keypoint_colors or INFERENCE_STYLE.keypoint_colors,
            font_scale=input_height / 960.0 * config['font_scale'],
            class_colors=self._class_colors,
            skeleton_colors=self._edge_colors,
            box_line_width=int(config['bbox_line_width']),
            skeleton_line_width=int(config['skeleton_line_width']),
            label_fill=True,
            label_text_color=(0, 0, 0),
        )

    def _infer_outputs(self, img):
        """Run the engine and hand back its *named* outputs.

        Named rather than positional: the decoder looks tensors up by name, which is what lets
        an artifact whose output is called "output0" decode at all.
        """
        engine = self.session.engine
        return engine.inference_named_outputs(img, output_names=engine.output_names or None)

    def _decode(self, outputs, inverse_affine):
        """Decode one frame's raw outputs into instances.

        Returns a dict so the postprocess thread can keep stamping its stage timings into it.
        """
        # Read live: the trio on the right is meant to be turned while the stream is running,
        # so the thresholds are re-applied each frame instead of using the snapshot taken at
        # Start.
        config = replace(
            self.session.output_config,
            confidence_threshold=self._live_poser_conf,
            iou_threshold=self._live_nms_iou,
        )
        return {
            "instances": decode_pose_outputs(outputs, config, inverse_affine=inverse_affine)
        }

    def _pass_frame_through(self, frame):
        """Two-stage: the frame reaches the pipeline whole; the detector does the cutting."""
        return [frame, None]

    def _infer_topdown(self, frame):
        """Two-stage: one frame in, decoded instances out.

        The live boxes ride along: the detector's threshold is written onto the adapter and
        the decode values onto a fresh output_config each frame, so turning them mid-run is
        what the next frame reads.
        """
        self._stream_pipeline.detector.conf_threshold = self._live_detector_conf
        self._stream_pipeline.output_config = replace(
            self._stream_pipeline.output_config,
            confidence_threshold=self._live_poser_conf,
            iou_threshold=self._live_nms_iou,
        )
        return self._stream_pipeline.process_frame(frame)

    def _wrap_instances(self, instances, _inverse_affine):
        """Two-stage decodes inside the pipeline; just carry the instances onward."""
        return {"instances": instances}

    def _draw_background(self, frame):
        """The Background selector's three states, applied before anything else is drawn."""
        mode = self.visualization_config['background']
        if mode == 'Black':
            return np.zeros_like(frame)
        if mode == 'White':
            return np.ones_like(frame) * 255
        return frame

    def _draw_timings(self, frame, results):
        """FPS and the three stage timings, top-left, in the order the old renderer used."""
        config = self.visualization_config

        def milliseconds(key):
            return f"{float(results.get(key, 0.0)) * 1000.0:.3f} ms"

        lines = []
        if config['show_fps']:
            # Clamped: the value is derived from a measured stage time, and an outlier used to
            # print as a seven-digit frame rate.
            fps_value = min(max(float(results.get('fps', 0.0)), 0.0), 999.9)
            lines.append(f"FPS: {fps_value:.1f}")
        if config['show_preprocess_time']:
            lines.append(f"Preprocess: {milliseconds('preprocess_time')}")
        if config['show_inference_time']:
            lines.append(f"Inference: {milliseconds('inference_time')}")
        if config['show_postprocess_time']:
            lines.append(f"Postprocess: {milliseconds('postprocess_time')}")
        if not lines:
            return frame

        style = self._render_style(frame)
        input_width, input_height = self.session.input_config.input_size
        x, y = int(input_width * 0.02), int(input_height * 0.05)
        # Only the drawn lines advance the cursor, exactly as before: skipping one pulls the
        # rest of the block up instead of leaving a hole.
        step = int(input_height * 0.04 * config['font_scale'])
        thickness = max(1, int(style.font_scale))
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            y += step
        return frame

    def _present(self, frame, results):
        """Render one decoded frame, and keep the frame's records for the caller."""
        frame = self._draw_background(frame)
        # Built once per run (see `_build_palettes`) rather than per frame: the session's class
        # names and skeleton do not change while a run is live.
        style = self._run_style
        instances = list(results.get("instances") or ())
        class_names = self._run_class_names
        records = [instance_to_record(instance, class_names) for instance in instances]
        if self.tracker is not None:
            # Tracking is a stage *on the records*, not on the pixels: the same shape comes out,
            # each detection carrying a track_id. That is what lets the offline workbench replay
            # the same records later with another tracker.
            from animalposetracker.postprocessing.workflows import track_frame_predictions

            records = track_frame_predictions(
                records,
                self.tracker,
                self._frame_index,
                frame=frame,
                class_names=class_names,
            )
        results["records"] = records
        # Every frame is kept: the records are this run's result and they are what gets written
        # out at the end, so there is nothing to ask for first.
        self.records.append({"frame": self._frame_index, "predictions": records})
        self._frame_index += 1

        config = self.visualization_config
        skeleton = self._run_skeleton
        for record in records:
            label = None
            if config['show_classes']:
                label = f"{record['class_name']}: {float(record.get('score', 0.0)):.2f}"
                track_id = record.get("track_id")
                if track_id is not None:
                    label = f"#{track_id} {label}"
            draw_instance(
                frame,
                pixel_box(record['bbox_xyxy']),
                label,
                pixel_points(record['keypoints']),
                skeleton,
                style=style,
                class_id=record['class_id'],
                show_box=config['show_bbox'],
                show_label=bool(label),
                show_keypoints=config['show_keypoints'],
                show_skeleton=config['show_skeletons'],
                line_width=2,
                point_radius=int(config['radius']),
            )
        self._draw_timings(frame, results)
        return frame, results

    def _init_inference_threads(self):
        source = self._get_source()
        if source is None:
            self.videoreader_thread = None
            return
        
        # set up cache queues
        self.read_cache = queue.Queue(maxsize=self.max_cache_size)
        self.preprocess_cache = queue.Queue(maxsize=self.max_cache_size)
        self.inference_cache = queue.Queue(maxsize=self.max_cache_size)
        self.postprocess_cache = queue.Queue(maxsize=self.max_cache_size)
        self.visualize_cache = queue.Queue(maxsize=self.max_cache_size)

        self.videoreader_thread = VideoReaderThread()
        self.videowriter_thread = VideoWriterThread()
        self.preprocess_thread = PreprocessThread(input_queue=self.read_cache)
        self.inference_thread = InferenceThread(input_queue=self.preprocess_cache) 
        self.postprocess_thread = PostprocessThread(input_queue=self.inference_cache)
        self.visualize_thread = VisualizeThread(input_queue=self.postprocess_cache)
        self.display_thread = QTimer()
        self.read_frame_end = False

        # Set up video reader thread
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            # A warning alone used to be followed by starting the reader against a handle that
            # never opened, so the run produced nothing and looked like it had worked.
            QMessageBox.warning(self, "Inference", "Failed to open camera/video source")
            self.videoreader_thread = None
            return
        if self.CameraORVideos.isChecked():
            self._set_camera_params(cap)
        else:
            # A video knows its own frame rate; the spin box is only what a camera was asked for.
            # Without this the display timer divided by a frame rate that was never set.
            self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.videoreader_thread.cap = cap
        self.videoreader_thread.status_update.connect(self.thread_status)
        self.videoreader_thread.data_ready.connect(self.put_read_data)
        self.videoreader_thread.finished.connect(self.read_end)

        # Set up the three middle stages. One-stage keeps the engine's own preprocess and the
        # page's decode. Two-stage hands the whole frame to the shared stream pipeline
        # (detector -> crops -> SimCC), which decodes on the way through, so the other two
        # stages become pass-throughs and all the stage time lands under "Inference" -- the
        # pipeline does not report its internal split.
        if self._stream_pipeline is not None:
            self.preprocess_thread.preprocess_function = self._pass_frame_through
            self.inference_thread.inference_function = self._infer_topdown
            self.postprocess_thread.postprocess_function = self._wrap_instances
        else:
            # The engine's preprocess is the one the shared decoder's inverse affine belongs
            # to, which is why the CLI's pipeline calls it too.
            self.preprocess_thread.preprocess_function = self.session.engine.preprocess
            self.inference_thread.inference_function = self._infer_outputs
            self.postprocess_thread.postprocess_function = self._decode
        self.preprocess_thread.status_update.connect(self.thread_status)
        self.preprocess_thread.data_ready.connect(self.put_preprocessed_data)
        self.inference_thread.status_update.connect(self.thread_status)
        self.inference_thread.data_ready.connect(self.put_inference_data)
        self.postprocess_thread.status_update.connect(self.thread_status)
        self.postprocess_thread.data_ready.connect(self.put_postprocessed_data)

        # Set up visualization thread
        self.visualize_thread.visualize_function = self._present
        self.visualize_thread.data_ready.connect(self.put_visualized_data)
        self.visualize_thread.status_update.connect(self.thread_status)

        # Set up display thread
        self.display_thread.timeout.connect(self.display_inferece_frame)

        

        # set up video writer thread
        if self.Save.isChecked():
            save_path = Path.cwd() / "runs" / "inference"
            save_path.mkdir(exist_ok=True, parents=True)
            self.videowriter_thread.save_path = save_path / "output.avi"
            # A source that reports no frame rate (or a page where nothing was ever set) would
            # otherwise hand the writer a rate it cannot open a file with.
            self.videowriter_thread.fps = self.fps or 30.0
            # The writer sizes itself from the frames it receives; the spin box values are only
            # what the source was asked for, which a camera need not honour.
            self.write_frame_end = False
            self.videowriter_thread.finished.connect(self.write_end)
        else:
            self.write_frame_end = True


    def _start_inference_threads(self):
        self.videoreader_thread.start()
        self.preprocess_thread.start()
        self.inference_thread.start()
        self.postprocess_thread.start()
        self.visualize_thread.start()
        # A source that reports no frame rate leaves self.fps at zero; the display still has to
        # tick, so fall back to roughly thirty frames a second instead of dividing by zero. A
        # negative rate is truthy and produced a negative interval, which Qt rejects outright --
        # the timer never fired and the run never finalized.
        interval = int(1000 / self.fps) if self.fps and self.fps > 0 else 33
        self.display_thread.start(interval)
        if self.Save.isChecked():
            self.videowriter_thread.start()
    
    def check_threads_existence(self, attribute_name):
        """Check if the thread with the given attribute name exists"""
        if hasattr(self, attribute_name):
            if getattr(self, attribute_name) is not None and getattr(self, attribute_name).isRunning():
                return True
        return False

    def _stop_inference_threads(self):
        """Stop all inference threads and clean up resources"""
        # Define all threads with their associated signals to disconnect
        threads = [
            "videoreader_thread",
            "preprocess_thread",
            "inference_thread",
            "postprocess_thread",
            "visualize_thread",
        ]

        # Stop all threads and disconnect their signals
        for thread in threads:
            if self.check_threads_existence(thread):
                getattr(self, thread).safe_stop()  # Request thread termination
                getattr(self, thread).status_update.disconnect()
                getattr(self, thread).data_ready.disconnect()
            
        if (self.Save.isChecked() and 
            hasattr(self, "videowriter_thread") and 
            getattr(self, "videowriter_thread") is not None and
            getattr(self, "videowriter_thread").isRunning()):
            # `safe_stop` sets this too; setting it here as well means the writer is told to
            # finish even if it is reached through a different path.
            getattr(self, "videowriter_thread").stop_flag = True
            getattr(self, "videowriter_thread").safe_stop()

        if self.display_thread is not None:
            self.display_thread.stop()

        # Clear all cache queues
        queues = [
            'read_cache',
            'preprocess_cache',
            'inference_cache',
            'postprocess_cache',
            'visualize_cache'
        ]
        for cache in queues:
            if hasattr(self, cache):
                with getattr(self, cache).mutex:
                    getattr(self, cache).queue.clear()
                    getattr(self, cache).unfinished_tasks = 0
                    getattr(self, cache).all_tasks_done.notify_all()
            
    def _get_source(self):
        if self.CameraORVideos.isChecked():
            try:
                current_text = self.CameraVideosSelection.currentText()
                cam_index = self.camera_list[current_text]
                return cam_index
            except:
                QMessageBox.warning(self, "Camera", "Failed to parse camera index")
                return None
        else:
            if hasattr(self, 'current_video_path'):
                return self.current_video_path
            QMessageBox.warning(self, "Video", "No video file selected")
            return None
    
    def thread_status(self, message):
        """Show a worker message on the page as well as the console.

        It only printed before, and a packaged GUI has no console: a camera that failed to
        open, a worker error or a written records file was invisible there.
        """
        print(message)
        self.StatusLabel.setText(message)

    def write_end(self):
        """Handle when video writer thread has finished"""
        self.write_frame_end = True
    
    def read_end(self):
        """Handle when video reader thread has finished"""
        self.read_frame_end = True
    
    def put_read_data(self, frame):
        """Put new frame from video reader thread to preprocess thread"""
        try:
            self.read_cache.put_nowait(frame)
        except queue.Full:
            pass
    
    def put_preprocessed_data(self, frame, img, IM, pred_time):
        """Put new frame from preprocess thread to inference thread"""
        try:
            self.preprocess_cache.put_nowait((frame, img, IM, pred_time))
        except queue.Full:
            pass
    
    def put_inference_data(self, frame, pred, IM, pred_time, infer_time):
        """Put new frame from inference thread to postprocess thread"""
        try:
            self.inference_cache.put_nowait((frame, pred, IM, pred_time, infer_time))
        except queue.Full:
            pass
    
    def put_postprocessed_data(self, frame, results):
        """Put new frame from postprocess thread to visualization thread"""
        try:
            self.postprocess_cache.put_nowait((frame, results))
        except queue.Full:  
            pass

    def put_visualized_data(self, frame, results):
        """Put new frame from visualization thread to display"""
        try:
            self.visualize_cache.put_nowait((frame, results))
        except queue.Full:
            pass

    def display_inferece_frame(self):
        """Handle new frame from camera/video source"""
        if self.visualize_cache.empty() and not self.read_frame_end:
            return
        elif self.visualize_cache.empty() and self.read_frame_end:
            if hasattr(self, 'videowriter_thread') and self.videowriter_thread is not None:
                self.videowriter_thread.stop_flag = True
            if self.write_frame_end:
                self._stop_all_threads()
            return
        
        frame, _ = self.visualize_cache.get()

        if (self.Save.isChecked() and 
            self.videowriter_thread is not None 
            and self.videowriter_thread.isRunning()):
            self.videowriter_thread.add_frame(frame)
        
        if self.Show.isChecked():
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = RGB.shape
            QImg = QImage(RGB.data, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg).scaled(
                self.Display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
                )
            
            # One path, not two: this slot already runs on the GUI thread, so the queued
            # `invokeMethod` that followed re-set the same pixmap and kept a second copy of
            # every frame alive for nothing.
            self.Display.setPixmap(pixmap)
        else:
            self.Display.clear()

    def onWidthSetupChanged(self, value):
        """Handle changes to the Width slider value"""
        self.width = value
    
    def onHeightSetupChanged(self, value):
        """Handle changes to the Height slider value"""
        self.height = value

    def onFPSSetupChanged(self, value):
        """Handle changes to the FPS slider value"""
        self.fps = value
    
    def _set_camera_params(self, cap=None):
        """Set up camera parameters"""
        if cap is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
    
    def onModelBitsChanged(self):
        """Handle changes to the Model Bits combo box value"""
        self.model_bits = self.ModelBits.currentText()
    
    def onEngineSelectionChanged(self):
        """Handle changes in engine selection"""
        self.engine = self.EngineSelection.currentText()
        if self.engine == "":
            return
        self.DeviceSelection.clear()
        self.DeviceSelection.addItems(self.supported_engines_and_devices[self.engine])

    def _engine_key(self):
        """The builder's key for the engine the user picked.

        `build_inference_session` takes the lowercase key the command line passes (`"onnx"`),
        while the combo shows the display label (`"ONNX"`). Handing the label straight to the
        builder raised `KeyError: 'ONNX'` for every run, and the page's own error dialog then
        blamed the model. The reverse lookup keeps the two vocabularies in step.
        """
        label = self.EngineSelection.currentText() or self.engine
        for key, name in ENGINE_NAMES.items():
            if name == label:
                return key
        return (label or "").lower()
    
    def onDeviceSelectionChanged(self):
        """Handle changes in device selection"""
        self.device = self.DeviceSelection.currentText()
        if self.device == "":
            return
    
    def onStartClicked(self):
        """Handle when the Start button is clicked to begin processing"""
        # A run flag, not the button's caption: `_refresh_start_state` can re-light Start while a
        # run is live, and starting again overwrote the thread references, leaving the first
        # run's threads with nothing pointing at them.
        if self._run_active:
            return
        if self.Start.text() == "Start":
            source = self._get_source()
            if source is None:
                QMessageBox.warning(self, "Warning", "Please select a source")
                return
            if not self.weights_path or not self.data_config_path:
                # The builder indexes both paths; without them it failed with a raw TypeError
                # that the dialog then showed verbatim.
                QMessageBox.warning(self, "Cannot start inference", "Select the configuration file and the weights first.")
                return
            self._init_visualization_config()
            # One call assembles the artifact, the schema and the decode contract -- the same
            # call the `infer` command line makes, so both decode identically. In two-stage the
            # same call is told a detector exists, which is what unlocks the SimCC schema, and
            # the frame loop is handed the shared stream pipeline below.
            try:
                self.session = build_inference_session(
                    weights_path=self.weights_path,
                    config_path=self.data_config_path,
                    engine=self._engine_key(),
                    device=self.device,
                    # Straight from the combo: the handler that copies it into `model_bits` only
                    # runs if the user touched it, and this used to read the attribute instead.
                    bits=self.ModelBits.currentText(),
                    # Read straight from the controls: they are the source, and
                    # _init_visualization_config has just copied them into its dict.
                    conf=self.Conf.value(),
                    iou=self.IoU.value(),
                    topdown=self._two_stage,
                )
                self._stream_pipeline = None
                if self._two_stage:
                    from animalposetracker.inference.detector_adapter import OnnxDetectorAdapter
                    from animalposetracker.inference.stream_inference import StreamInferencePipeline

                    detector = OnnxDetectorAdapter(
                        Path(self.detector_path),
                        conf_threshold=self._live_detector_conf,
                        **self._detector_conventions(self.detector_path),
                    )
                    # The one place that owns detect -> crop -> decode for a stream, shared
                    # with the `infer` command line; the page only supplies the frames.
                    self._stream_pipeline = StreamInferencePipeline(
                        self.session.engine,
                        self.session.output_config,
                        input_mode="topdown",
                        input_config=self.session.input_config,
                        detector=detector,
                        class_names=list(self.session.class_names),
                    )
            except Exception as error:
                # Whatever the artifact and the config cannot support together (an unknown
                # keypoint count, a SimCC artifact, an unreadable file) is reported here
                # instead of surfacing as a numpy failure inside a worker thread.
                QMessageBox.critical(self, "Cannot start inference", f"Cannot start inference:\n{error}")
                return
            self._build_palettes()
            # One run, one buffer: the frame indices the records carry start at zero again.
            self._frame_index = 0
            self.records = []
            self._build_tracker()
            # init threads
            self._init_inference_threads()
            if self.videoreader_thread is None:
                # The source could not be opened; `_init_inference_threads` already said so.
                return
            self._run_active = True
            # start threads
            self._start_inference_threads()
            self.Start.setEnabled(False)
    
    def onEndClicked(self):
        """Handle when the End button is clicked to stop processing"""
        self._stop_all_threads()
    
    def _stop_all_threads(self):
        """Stop all threads and write this run's records."""
        if not hasattr(self, '_stopping'):
            self._stopping = True
            self._stop_inference_threads()
            # Written here, not from a deferred timer: the page is closed with `deleteLater()`
            # when the user switches tools, and a `singleShot` callback queued behind that ran
            # against a destroyed object, so the run's records were silently never saved.
            self._finalize_cleanup()

    def _finalize_cleanup(self):
        """Clean up after all threads have stopped"""
        try:
            # Every worker has stopped by now (see _stop_all_threads), so the record buffer is
            # final; the file is only useful whole, so it is written here rather than per frame.
            self._write_records()
            # The detector session belongs to the run that just ended; drop it with the run.
            self._stream_pipeline = None
            self.Display.clear()
            self.Start.setText("Start")
            self._run_active = False
            # Re-ask rather than enabling blindly: a run that ended with no valid selection used
            # to leave Start lit for a pair that cannot run.
            self._refresh_start_state(show_dialog=False)
        finally:
            if hasattr(self, '_stopping'):
                del self._stopping

    
    # Every display option below just writes the config the renderer reads each frame. They used
    # to push into the engine as well, which stopped meaning anything once the page took over
    # drawing.
    def onIoUChanged(self, value):
        """Handle changes to the IOU threshold slider value"""
        self.visualization_config.update({"iou": value})
        self._live_nms_iou = float(value)

    def onConfChanged(self, value):
        """Handle changes to the Confidence threshold slider value"""
        self.visualization_config.update({"conf": value})
        self._live_poser_conf = float(value)

    def onBackgroudStateChanged(self, state):
        """Handle changes to the Show Backgroud checkbox state"""
        if state == 0:
            background = "Original"
        elif state == 1:
            background = "White"
        else:
            background = "Black"
        self.Backgroud.setText(background)
        self.visualization_config.update({"background": background})

    def onShowClassesStateChanged(self, state):
        """Handle changes to the Show Classes checkbox state"""
        self.visualization_config.update({"show_classes": bool(state)})

    def onShowKeypointsStateChanged(self, state):
        """Handle changes to the Show Keypoints checkbox state"""
        self.visualization_config.update({"show_keypoints": bool(state)})

    def onShowKeypointsRadiusChanged(self, value):
        """Handle changes to the keypoint radius slider value"""
        self.visualization_config.update({"radius": value})

    def onShowSkeletonsStateChanged(self, state):
        """Handle changes to the Show Skeletons checkbox state"""
        self.visualization_config.update({"show_skeletons": bool(state)})

    def onShowSkeletonsLineWidthChanged(self, value):
        """Handle changes to the skeleton line width slider value"""
        self.visualization_config.update({"skeleton_line_width": value})

    def onShowBBoxStateChanged(self, state):
        """Handle changes to the Show Bounding Box checkbox state"""
        self.visualization_config.update({"show_bbox": bool(state)})

    def onShowBBoxWidthChanged(self, value):
        """Handle changes to the bounding box width slider value"""
        self.visualization_config.update({"bbox_line_width": value})

    def onFPSShowStateChanged(self, state):
        """Handle changes to the Show FPS checkbox state"""
        self.visualization_config.update({"show_fps": bool(state)})

    def onPreprocessTimeStateChanged(self, state):
        """Handle changes to the Show Preprocess Time checkbox state"""
        self.visualization_config.update({"show_preprocess_time": bool(state)})

    def onInferenceTimeStateChanged(self, state):
        """Handle changes to the Show Inference Time checkbox state"""
        self.visualization_config.update({"show_inference_time": bool(state)})

    def onPostprocessTimeStateChanged(self, state):
        """Handle changes to the Show Postprocess Time checkbox state"""
        self.visualization_config.update({"show_postprocess_time": bool(state)})

    def onFontScaleChanged(self, value):
        """Handle changes to the Font Scale slider value"""
        self.visualization_config.update({"font_scale": value / 10.0})
    
    def closeEvent(self, event):
        """Handle window close event"""
        self._stop_all_threads()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = InferencerPage()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()