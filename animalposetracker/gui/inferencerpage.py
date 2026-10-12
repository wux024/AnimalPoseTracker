from PySide6.QtCore import QMetaObject, Qt, Q_ARG, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  (QApplication, QFileDialog, QWidget, QMessageBox,
                                QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                                QSpinBox)
from dataclasses import replace
import json
import os
import cv2
import numpy as np
import sys
from pathlib import Path
import queue

from .ui_inferencer import Ui_Inferencer
from animalposetracker.utils import (VideoReaderThread, VideoWriterThread, 
                                     PreprocessThread, InferenceThread, 
                                     PostprocessThread, VisualizeThread)
from animalposetracker.inference import ENGINEtoDEVICE
from animalposetracker.inference.session import (
    build_inference_session,
    instance_to_record,
    pixel_points,
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
        self.video_reader_thread = None
        self.video_writer_thread = None
        self.preprocess_thread = None
        self.inference_thread = None
        self.postprocess_thread = None
        self.visualize_thread = None
        self.write_frame_end = True

        self.data_config_path = None
        self.weights_path = None
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
        self._add_record_switch()
        # The filter's rows go in first because they sit above the tracker's in the group.
        self._add_filter_controls()
        self._add_tracker_controls()

        self.platform = self._detect_platform()
        self.device_check_results = self._detect_device_availability()

        self.tools_disabled()
        self.SelectConfigure.setEnabled(True)
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
        self.SelectWeights.clicked.connect(self.onSelectWeightsClicked)
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
        self.Start.setEnabled(True)
        self.End.setEnabled(True)
        self.Save.setEnabled(True)
        self.IoU.setEnabled(True)
        self.Conf.setEnabled(True)
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
        self.SaveRecords.setEnabled(True)
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
        self.IoU.setEnabled(False)
        self.Conf.setEnabled(False)
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
        self.SaveRecords.setEnabled(False)
        self.EnableTracking.setEnabled(False)
        self.TrackerAlgorithm.setEnabled(False)
        for widget in self._filter_controls():
            widget.setEnabled(False)
    def _add_record_switch(self):
        """Add the "save records" switch to the display options.

        Built here rather than in the Designer file for the same reason the parameter trees are
        shaped at runtime: ``ui_inferencer.py`` is generated, so a widget put there would be lost
        the next time the file is regenerated.
        """
        # "Save" alone read as ambiguous next to "Save Records". Each switch now says what it
        # saves; the label is set here because ui_inferencer.py is generated.
        self.Save.setText("Save Video")
        self.Save.setToolTip("把渲染后的画面写成 runs/inference/output.avi")

        self.SaveRecords = QCheckBox(self.ShowToolsGroup)
        self.SaveRecords.setObjectName("SaveRecords")
        self.SaveRecords.setMinimumSize(QSize(120, 25))
        self.SaveRecords.setMaximumSize(QSize(120, 25))
        self.SaveRecords.setText("Save Records")
        self.SaveRecords.setToolTip(
            "存的是逐帧姿态记录（不是画面）：runs/inference/records.json，"
            "形状与命令行 `infer --records` 一致，给跟踪/轨迹/3D 重建用"
        )
        self.ShowToolsGroupVLayout.addWidget(self.SaveRecords)

    def _add_tracker_controls(self):
        """Add the online-tracking switch and its algorithm picker.

        The list comes from the same tuple the `--tracker` command line validates against,
        imported here rather than at module scope: the tracking package costs about a second to
        import, and the page would pay it every time the window opens.
        """
        from animalposetracker.tracking.config import ALGORITHMS

        self.EnableTracking = QCheckBox(self.ShowToolsGroup)
        self.EnableTracking.setObjectName("EnableTracking")
        self.EnableTracking.setText("Track")
        self.EnableTracking.setToolTip(
            "逐帧跟踪目标：画面标签会带上 track ID，保存的记录里也会带 track_id"
        )
        self.TrackerAlgorithm = QComboBox(self.ShowToolsGroup)
        self.TrackerAlgorithm.setObjectName("TrackerAlgorithm")
        self.TrackerAlgorithm.setMinimumSize(QSize(0, 25))
        self.TrackerAlgorithm.setMaximumSize(QSize(88, 25))
        self.TrackerAlgorithm.addItems(list(ALGORITHMS))
        self.TrackerAlgorithm.setToolTip("跟踪算法，与命令行 `infer --tracker` 是同一份清单")
        self.TrackerAlgorithm.setEnabled(False)

        # The Designer file leaves an empty TrackerLayout inside the left group's engine/device
        # block. The tracking row belongs with the output switches instead, and Qt refuses to
        # insert a layout that already has a parent -- so the placeholder is dropped and a fresh
        # layout takes the name.
        placeholder = self.TrackerLayout
        self.EngineDeviceLayout.removeItem(placeholder)
        self.TrackerLayout = QHBoxLayout()
        self.TrackerLayout.setObjectName("TrackerLayout")

        self.TrackerLayout.addWidget(self.EnableTracking)
        self.TrackerLayout.addWidget(self.TrackerAlgorithm)
        # Third row of the block at the top of the right-hand group: the filter's two rows above,
        # then what the filter needs (track ids) to have anything to work on.
        self.ShowToolsGroupVLayout.insertLayout(2, self.TrackerLayout)

    def _add_filter_controls(self):
        """Add the temporal filter's controls, on the first two rows of the right-hand group.

        Every switch in that group is a checkbox, so the filter gets one of its own: it is the
        on/off, which is why the method list has no `none` -- one way to be off is enough. The
        values and defaults match `infer --pose-filter` and its `--filter-*` arguments, so a look
        tuned here can be written down as a command line and the other way round.

        The smoothing acts on a whole sequence and therefore runs when the run ends; it changes
        the records file, never the live picture.
        """
        self.FilterSwitchLayout = QHBoxLayout()
        self.FilterSwitchLayout.setObjectName("FilterSwitchLayout")

        self.EnableFilter = QCheckBox(self.ShowToolsGroup)
        self.EnableFilter.setObjectName("EnableFilter")
        self.EnableFilter.setText("Filter")
        self.EnableFilter.setToolTip("跑完对整段轨迹平滑一次：只改存下来的记录，不改实时画面")
        self.FilterSwitchLayout.addWidget(self.EnableFilter)

        self.FilterMethod = QComboBox(self.ShowToolsGroup)
        self.FilterMethod.setObjectName("FilterMethod")
        self.FilterMethod.setMaximumSize(QSize(88, 25))
        self.FilterMethod.addItems(["median", "savgol"])
        self.FilterMethod.setToolTip("median = 中值平滑；savgol = Savitzky-Golay 拟合")
        self.FilterSwitchLayout.addWidget(self.FilterMethod)

        self.FilterValuesLayout = QHBoxLayout()
        self.FilterValuesLayout.setObjectName("FilterValuesLayout")

        def spin(widget, name, prefix, value, maximum, tooltip):
            widget.setObjectName(name)
            widget.setPrefix(f"{prefix} ")
            widget.setRange(widget.minimum(), maximum)
            widget.setValue(value)
            widget.setToolTip(tooltip)
            widget.setMaximumSize(QSize(62, 25))
            self.FilterValuesLayout.addWidget(widget)
            return widget

        self.FilterWindow = spin(
            QSpinBox(self.ShowToolsGroup), "FilterWindow", "win", 5, 999, "平滑窗口帧数"
        )
        self.FilterPolyorder = spin(
            QSpinBox(self.ShowToolsGroup), "FilterPolyorder", "poly", 2, 99,
            "Savitzky-Golay 多项式阶数（仅 savgol 用）",
        )
        self.FilterConfidence = spin(
            QDoubleSpinBox(self.ShowToolsGroup), "FilterConfidence", "conf", 0.25, 1.0,
            "低于此分数的关键点不参与平滑",
        )
        self.FilterConfidence.setSingleStep(0.05)
        self.FilterConfidence.setDecimals(2)
        self.FilterMaxGap = spin(
            QSpinBox(self.ShowToolsGroup), "FilterMaxGap", "gap", 0, 999,
            "允许插值的最大缺失帧数（0 = 不插）",
        )

        # Above the tracker's row, and at the top of the group: the switch that turns the filter
        # on, then its values, then what the filter needs (tracking).
        self.ShowToolsGroupVLayout.insertLayout(0, self.FilterSwitchLayout)
        self.ShowToolsGroupVLayout.insertLayout(1, self.FilterValuesLayout)
        for widget in self._filter_controls():
            widget.setEnabled(False)

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
        result, and the result is what the filter produced.
        """
        if not self.SaveRecords.isChecked() or not self.records:
            return
        smoothing = self._filter_is_on()
        try:
            records = self._final_records()
        except (ValueError, KeyError) as error:
            # Records without track ids are refused by the workflow. The controls already prevent
            # that pairing, so this is the belt to their braces -- and it must not cost a whole
            # run's records on the way out.
            QMessageBox.warning(self, "整段平滑未执行", str(error))
            records = self.records

        output_dir = Path.cwd() / "runs" / "inference"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "records.json"
            target.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as error:
            QMessageBox.warning(self, "保存记录失败", f"无法写入逐帧记录：\n{error}")
            return
        note = "（已平滑）" if smoothing else ""
        self.thread_status(f"逐帧记录已写入 {target}{note}（{len(records)} 帧）")

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
        
        # set home directory
        os.chdir(Path(file_path).parent)
            
        # Store configuration
        self.data_config_path = file_path
  
        # Update UI
        self.SelectWeights.setEnabled(True)
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
        report = validate_inference_selection(self.data_config_path, self.weights_path)
        self._config_ready = report.config_ok
        self._weights_ready = report.weights_ok
        self.Start.setEnabled(self._config_ready and self._weights_ready)

        for warning in report.warnings:
            self.thread_status(f"警告：{warning}")
        if report.errors and show_dialog:
            detail = "\n".join(report.errors)
            if report.warnings:
                detail += "\n\n另外注意：\n" + "\n".join(report.warnings)
            QMessageBox.critical(self, "配置校验失败", detail)
            

    def onSelectWeightsClicked(self):
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

        self.weights_path = file_path
        # Check before probing the hardware: a pair that cannot run should say so now, and the
        # probe is the expensive part of this slot.
        report = validate_inference_selection(self.data_config_path, self.weights_path)
        if not report.weights_ok:
            self._config_ready = report.config_ok
            self._weights_ready = False
            self.Start.setEnabled(False)
            self._clear_selections()
            QMessageBox.critical(self, "配置校验失败", "\n".join(report.errors))
            return

        self._enable_widgets()
        self._clear_selections()
        self.supported_engines_and_devices = {}

        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        

        self._get_supported_engines_and_devices(file_ext)

        if not self.supported_engines_and_devices:
            raise ValueError(f"Unsupported format: {file_ext}")

        self._init_engine_selection(list(self.supported_engines_and_devices.keys()))
        self._init_available_devices()
        self._init_model_bits()

        # Start needs both halves of the configuration. This used to test an attribute that is
        # never assigned anywhere, so it stayed disabled until a camera or video was touched.
        self._refresh_start_state(show_dialog=False)

    def _enable_widgets(self):
        self.EngineSelection.setEnabled(True)
        self.DeviceSelection.setEnabled(True)
        self.CameraORVideos.setEnabled(True)
        self.CheckCameraVideosConnect.setEnabled(True)

    def _clear_selections(self):
        self.EngineSelection.clear()
        self.DeviceSelection.clear()
    
    def _get_supported_engines_and_devices(self, file_ext):
        """Get supported engines based on file extension"""
        if file_ext == '.onnx':
            supported_engines = ["ONNX", "OpenVINO", "TensorRT", "OpenCV"]
        elif file_ext == '.xml':
            supported_engines =  ["OpenVINO"]
        elif file_ext == '.om':
            supported_engines =  ["CANN", "OpenCV"]
        elif file_ext == '.engine':
            supported_engines =  ["TensorRT"]
        elif file_ext in ['.mlmodel', '.mlpackage']:
            supported_engines =  ["CoreML"]
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        for engine in supported_engines:
            device_list = self._get_supported_devices(engine)
            if device_list:
                self.supported_engines_and_devices[engine] = device_list
                
    def _init_engine_selection(self, engines):
        self.EngineSelection.addItems(engines)
        default_engine = engines[0]
        self.EngineSelection.setCurrentText(default_engine)

    def _detect_platform(self):
        """Detect platform and hardware availability"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            vendor = info.get('vendor_id_raw', '')
            if vendor == 'GenuineIntel':
                return 'Intel'
            elif vendor == 'AuthenticAMD':
                return 'AMD'
            elif 'ARM' in info.get('arch', ''):
                return 'ARM'
            else:
                raise ValueError(f"Unsupported CPU vendor: {vendor}")
        except ImportError:
            raise ImportError("Please install 'py-cpuinfo' module")
    
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

    def _check_intel_gpu(self):
        """Check if Intel GPU is available in the system"""
        try:
            import subprocess
            if os.name == 'nt': 
                cmd = 'wmic path win32_VideoController get name'
                result = subprocess.run(cmd, shell=True, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.DEVNULL,
                                        text=True, encoding='utf-8')
                return "Intel" in result.stdout
            else: 
                cmd = "lspci | grep -i 'VGA.*Intel'"
                result = subprocess.run(cmd, shell=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
                return result.returncode == 0
        except:
            return False
        
    def _check_intel_npu(self):
        """Check if Intel NPU is available through OpenVINO"""
        try:
            import subprocess
            if os.name == 'nt':  
                cmd = 'wmic path Win32_PnPEntity get name | findstr /i "AI Boost"'
                result = subprocess.run(cmd, shell=True, 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL,
                                    text=True)
                return "AI Boost" in result.stdout
            
            else:
                return os.path.exists("/dev/accel/accel0") or \
                    os.path.exists("/sys/class/accel/accel0")
        except:
            return False
    
    def _check_amd_gpu(self):
        """Check if AMD GPU is available in the system (Windows/Linux)"""
        try:
            import subprocess
            if os.name == 'nt': 
                cmd = 'wmic path win32_VideoController get name'
                result = subprocess.run(cmd, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL,
                                    text=True,
                                    encoding='utf-8')
                return "AMD" in result.stdout or "Radeon" in result.stdout
                
            else:
                cmd = "lspci | grep -i 'VGA.*AMD'"
                result = subprocess.run(cmd, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL,
                                    text=True)
                return result.returncode == 0 and len(result.stdout) > 0
        except:
            return False

    def _check_nvidia_gpu(self):
        """Check if NVIDIA GPU is available in the system"""
        try:
            import subprocess
            subprocess.run(['nvidia-smi'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True 
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _check_nvidia_gpu_tensorrt(self):
        """Check if NVIDIA GPU with TensorRT is available in the system"""
        try:
            if not self._check_nvidia_gpu():
                return False
            import tensorrt as trt
            return trt.__version__ is not None
        except ImportError:
            print("Please install 'tensorrt' module! "
                  "if NVIDIA GPU with TensorRT is available in the system!")
            return False

    def _check_ascend_npu(self):
        """Check if CANN environment is available for Ascend NPU"""
        try:
            import subprocess
            result = subprocess.run(
                ['npu-smi', 'info'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

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
        self.CameraORVideos.setText("Use Cramera")  # Ensure text matches design
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
            
        except Exception as e:
            print(f"Error updating camera params: {str(e)}")
            # Fallback to defaults
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
        self.videoreader_thread = VideoReaderThread()
        self.videoreader_thread.cap = cap
        self.videoreader_thread.data_ready.connect(self.display_preview_frame)
        self.videoreader_thread.finished.connect(self._stop_preview)
        self.videoreader_thread.start()
        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Close {mode}")

    def _stop_preview(self):
        """Stop camera/video preview"""
        self.videoreader_thread.safe_stop()
        self.Display.clear()
        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Preview {mode}")

    def display_preview_frame(self, frame):
        """Handle new frame from camera/video source"""

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

        Deliberately random, as the engine's own renderer always was: the colours are
        decoration, and nobody should read meaning into which class got which hue.
        """
        def palette(count):
            if count <= 0:
                return ()
            colours = np.random.uniform(0, 255, size=(count, 3))
            return tuple(tuple(int(channel) for channel in colour) for colour in colours)

        self._class_colors = palette(len(self.session.class_names))
        self._edge_colors = palette(len(self.session.skeleton))
        self._keypoint_colors = palette(self.session.kpt_shape[0])

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
        return {
            "instances": decode_pose_outputs(
                outputs, self.session.output_config, inverse_affine=inverse_affine
            )
        }

    def _render_style(self, frame):
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
            lines.append(f"FPS: {float(results.get('fps', 0.0)):.1f}")
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
        style = self._render_style(frame)
        instances = list(results.get("instances") or ())
        class_names = list(self.session.class_names)
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
        # Only collected when the switch asks for it: a camera session can run for hours, and an
        # unbounded buffer is not worth carrying for a file nobody asked for.
        if self.SaveRecords.isChecked():
            self.records.append({"frame": self._frame_index, "predictions": records})
        self._frame_index += 1

        config = self.visualization_config
        skeleton = list(self.session.skeleton)
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
        self.videowriter_thread = VideoWriterThread()
        self.display_thread = QTimer()
        self.read_frame_end = False

        # Set up video reader thread
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            QMessageBox.warning(self, "Inference", "Failed to open camera/video source")
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

        # Set up frame preprocessor thread. The engine's preprocess is the one the shared
        # decoder's inverse affine belongs to, which is why the CLI's pipeline calls it too.
        self.preprocess_thread.preprocess_function = self.session.engine.preprocess
        self.preprocess_thread.status_update.connect(self.thread_status)
        self.preprocess_thread.data_ready.connect(self.put_preprocessed_data)

        # Set up inference thread
        self.inference_thread.inference_function = self._infer_outputs
        self.inference_thread.status_update.connect(self.thread_status)
        self.inference_thread.data_ready.connect(self.put_inference_data)

        # Set up postprocessor thread
        self.postprocess_thread.postprocess_function = self._decode
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
        # tick, so fall back to roughly thirty frames a second instead of dividing by zero.
        self.display_thread.start(int(1000 / self.fps) if self.fps else 33)
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
            getattr(self, "videowriter_thread").safe_stop()

        if hasattr(self, "display_thread"):
            getattr(self, "display_thread").stop()

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
        print(message)

    def write_end(self):
        """Handle when video writer thread has finished"""
        self.write_frame_end = True
    
    def read_end(self):
        """Handle when video reader thread has finished"""
        self.read_frame_end = True
    
    def put_read_data(self, frame):
        """Put new frame from video reader thread to preprocess thread"""
        try:
            self.read_cache.put(frame)
        except queue.Full:
            pass
    
    def put_preprocessed_data(self, frame, img, IM, pred_time):
        """Put new frame from preprocess thread to inference thread"""
        try:
            self.preprocess_cache.put((frame, img, IM, pred_time))
        except queue.Full:
            pass
    
    def put_inference_data(self, frame, pred, IM, pred_time, infer_time):
        """Put new frame from inference thread to postprocess thread"""
        try:
            self.inference_cache.put((frame, pred, IM, pred_time, infer_time))
        except queue.Full:
            pass
    
    def put_postprocessed_data(self, frame, results):
        """Put new frame from postprocess thread to visualization thread"""
        try:
            self.postprocess_cache.put((frame, results))
        except queue.Full:  
            pass

    def put_visualized_data(self, frame, results):
        """Put new frame from visualization thread to display"""
        try:
            self.visualize_cache.put((frame, results))
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
            
            self.Display.setPixmap(pixmap)

            QMetaObject.invokeMethod(
                self.Display,
                "setPixmap",
                Qt.QueuedConnection,
                Q_ARG(QPixmap, pixmap)
            )
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
    
    def onDeviceSelectionChanged(self):
        """Handle changes in device selection"""
        self.device = self.DeviceSelection.currentText()
        if self.device == "":
            return
    
    def onStartClicked(self):
        """Handle when the Start button is clicked to begin processing"""
        if self.Start.text() == "Start":
            source = self._get_source()
            if source is None:
                QMessageBox.warning(self, "Warning", "Please select a source")
                return
            self._init_visualization_config()
            # One call assembles the artifact, the schema and the decode contract -- the same
            # call the `infer` command line makes, so both decode identically.
            try:
                self.session = build_inference_session(
                    weights_path=self.weights_path,
                    config_path=self.data_config_path,
                    engine=self.engine,
                    device=self.device,
                    # Straight from the combo: the handler that copies it into `model_bits` only
                    # runs if the user touched it, and this used to read the attribute instead.
                    bits=self.ModelBits.currentText(),
                    # Read straight from the controls: they are the source, and
                    # _init_visualization_config has just copied them into its dict.
                    conf=self.Conf.value(),
                    iou=self.IoU.value(),
                )
            except Exception as error:
                # Whatever the artifact and the config cannot support together (an unknown
                # keypoint count, a SimCC artifact, an unreadable file) is reported here, in
                # Chinese, instead of surfacing as a numpy failure inside a worker thread.
                QMessageBox.critical(self, "无法启动推理", f"无法启动推理：\n{error}")
                return
            self._build_palettes()
            # One run, one buffer: the frame indices the records carry start at zero again.
            self._frame_index = 0
            self.records = []
            self._build_tracker()
            # init threads
            self._init_inference_threads()
            # start threads
            self._start_inference_threads()
            self.Start.setEnabled(False)
    
    def onEndClicked(self):
        """Handle when the End button is clicked to stop processing"""
        self._stop_all_threads()
    
    def _stop_all_threads(self):
        """Stop all threads"""
        if not hasattr(self, '_stopping'):
            self._stopping = True
            self._stop_inference_threads()
            QTimer.singleShot(33, self._finalize_cleanup)
    
    def _finalize_cleanup(self):
        """Clean up after all threads have stopped"""
        try:
            # Every worker has stopped by now (see _stop_all_threads), so the record buffer is
            # final; the file is only useful whole, so it is written here rather than per frame.
            self._write_records()
            self.Display.clear()
            self.Start.setText("Start")
            self.Start.setEnabled(True)
        finally:
            if hasattr(self, '_stopping'):
                del self._stopping

    
    # Every display option below just writes the config the renderer reads each frame. They used
    # to push into the engine as well, which stopped meaning anything once the page took over
    # drawing.
    def onIoUChanged(self, value):
        """Handle changes to the IOU threshold slider value"""
        self.visualization_config.update({"iou": value})

    def onConfChanged(self, value):
        """Handle changes to the Confidence threshold slider value"""
        self.visualization_config.update({"conf": value})

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