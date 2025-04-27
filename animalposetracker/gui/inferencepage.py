from PySide6.QtCore import QMetaObject, Qt, Slot, Q_ARG, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget, QMessageBox
import os
import yaml
import cv2
import numpy as np
import sys
from pathlib import Path
import queue

from .ui_animalposeinference import Ui_AnimalPoseInference
from animalposetracker.utils import (VideoReaderThread, VideoWriterThread, 
                                     FrameProcessorThread)
from animalposetracker.engine import InferenceEngine
from animalposetracker.engine import ENGINEtoDEVICE
class AnimalPoseInferencePage(QWidget, Ui_AnimalPoseInference):
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
        self.original_frame_cache = queue.Queue(maxsize=1024)
        self.processed_frame_cache = queue.Queue(maxsize=1024)

        self.video_reader_thread = VideoReaderThread()
        self.video_writer_thread = VideoWriterThread()
        self.frame_processor_thread = FrameProcessorThread(buffer_queue=self.original_frame_cache)
        self.display_thread = QTimer()
        self.inference = InferenceEngine()
        self.data_config_path = None
        self.data_config = dict()
        self.weights_path = None

        self.platform = self._detect_platform()

        self.tools_disabled()
        self.SelectConfigure.setEnabled(True)
        self.ShowKeypoints.setCheckState(Qt.CheckState.Checked)
        self.Show.setCheckState(Qt.CheckState.Checked)
        self._show_info("Please select configuration and weights")

        # Connect all signals to their respective slots
        self.setupConnections()
        # Set default visualization configuration
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
        self.EngineSelection.currentIndexChanged.connect(self.onEngineSelectionChanged)
        self.DeviceSelection.currentIndexChanged.connect(self.onDeviceSelectionChanged)
        
        # Start/Stop control signals
        self.Start.clicked.connect(self.onStartClicked)
        self.End.clicked.connect(self.onEndClicked)
        
        # Display option signals
        self.Save.stateChanged.connect(self.onSaveStateChanged)
        self.IoU.valueChanged.connect(self.onIoUChanged)
        self.Conf.valueChanged.connect(self.onConfChanged)
        self.Show.stateChanged.connect(self.onShowStateChanged)
        self.Backgroud.stateChanged.connect(self.onBackgroudStateChanged)
        self.ShowClasses.stateChanged.connect(self.onShowClassesStateChanged)
        self.ShowKeypoints.stateChanged.connect(self.onShowKeypointsStateChanged)
        self.ShowKeypointsRadius.valueChanged.connect(self.onShowKeypointsRadiusChanged)
        self.ShowSkeletons.stateChanged.connect(self.onShowSkeletonsStateChanged)
        self.ShowSkeletonsLineWidth.valueChanged.connect(self.onShowSkeletonsLineWidthChanged)
        self.ShowBBox.stateChanged.connect(self.onShowBBoxStateChanged)
        self.ShowBBoxWidth.valueChanged.connect(self.onShowBBoxWidthChanged)

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
        self.Classes.setEnabled(True)
        self.ShowKeypoints.setEnabled(True)
        self.ShowKeypointsRadius.setEnabled(True)
        self.ShowSkeletons.setEnabled(True)
        self.ShowSkeletonsLineWidth.setEnabled(True)
        self.ShowBBox.setEnabled(True)
        self.ShowBBoxWidth.setEnabled(True)

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
        self.Classes.setEnabled(False)
        self.ShowKeypoints.setEnabled(False)
        self.ShowKeypointsRadius.setEnabled(False)
        self.ShowSkeletons.setEnabled(False)
        self.ShowSkeletonsLineWidth.setEnabled(False)
        self.ShowBBox.setEnabled(False)
        self.ShowBBoxWidth.setEnabled(False)
    
    def _init_visualization_config(self):
        """Initialize visualization configuration"""
        self.inference.update_config(
            {
            'conf': self.Conf.value(),
            'iou': self.IoU.value(),
           'show_classes': self.ShowClasses.isChecked(),
           'show_keypoints': self.ShowKeypoints.isChecked(),
           'show_skeletons': self.ShowSkeletons.isChecked(),
           'show_bbox': self.ShowBBox.isChecked(),
            'radius': self.ShowKeypointsRadius.value(),
           'skeleton_line_width': self.ShowSkeletonsLineWidth.value(),
            'bbox_line_width': self.ShowBBoxWidth.value(),
            'background': {
                0: 'Original',
                1: 'White',
                2: 'Black',
            }[self.Backgroud.checkState().value]
        })

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
            self._show_info("Configuration selection cancelled")
            return
            
        try:
            # Validate YAML structure
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Store configuration
            self.data_config_path = file_path
            self.data_config = config
            
            # Update UI
            filename = os.path.basename(file_path)
            self._show_info(f"Configuration loaded: {filename}")
            self.SelectWeights.setEnabled(True)
            
        except yaml.YAMLError as e:
            self._show_info(f"Invalid YAML: {str(e)}")
            self.data_config_path = None

        except Exception as e:
            self._show_info(f"Error: {str(e)}")
            self.data_config_path = None

    def onSelectWeightsClicked(self):
        """Open file dialog to select model weights and 
        auto-configure supported engines and devices.

        Supported Engines:
        - OpenVINO (.onnx and .xml)
        - OpenCV (.onnx)
        - CANN (.om)
        - ONNX (.onnx)
        - TensorRT (.engine)
        - CoreML (.mlmodel)

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
            "All Supported Formats (*.xml *.onnx *.om *.engine);;"
            "ONNX (*.onnx);;"
            "CANN Model (*.om);;"
            "TensorRT (*.engine);;"
            "OpenVINO (select .xml);;"
            "CoreML (*.mlmodel);;"
            "All Files (*)",
            options=options
        )

        if not file_path:
            self._show_info("Weight selection cancelled")
            return

        self._enable_widgets()
        self._clear_selections()

        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        self.weights_path = file_path

        engines = self._get_supported_engines(file_ext)

        if not engines:
            self._show_info("Unsupported format for current platform")
            raise ValueError(f"Unsupported format: {file_ext}")

        self._populate_engine_selection(engines)
        self._update_available_devices()

        if hasattr(self, 'model_config_path'):
            self.Start.setEnabled(True)

    def _enable_widgets(self):
        self.EngineSelection.setEnabled(True)
        self.DeviceSelection.setEnabled(True)
        self.CameraORVideos.setEnabled(True)
        self.CheckCameraVideosConnect.setEnabled(True)

    def _clear_selections(self):
        self.EngineSelection.clear()
        self.DeviceSelection.clear()
    
    def _get_supported_engines(self, file_ext):
        """Get supported engines based on file extension"""
        if file_ext == '.onnx':
            supported_engines = ["OpenCV", "ONNX", 'OpenVINO']
        elif file_ext == '.xml':
            supported_engines =  ["OpenVINO"]
        elif file_ext == '.om':
            supported_engines =  ["CANN"]
        elif file_ext == '.engine':
            supported_engines =  ["TensorRT"]
        elif file_ext == '.mlmodel':
            supported_engines =  ["CoreML"]
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        return supported_engines

    def _populate_engine_selection(self, engines):
        self.EngineSelection.addItems(engines)
        default_engine = engines[0]
        self.EngineSelection.setCurrentText(default_engine)

    def _show_info(self, message):
        self.PrintInformation.setText(message)

    def _detect_platform(self):
        """Detect platform and hardware availability"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            vendor = info.get('vendor_id_raw')
            if vendor == 'GenuineIntel':
                return 'Intel'
            elif vendor == 'AuthenticAMD':
                return 'AMD'
            elif 'ARM' in info.get('brand_raw', ''):
                return 'ARM'
            else:
                raise ValueError(f"Unsupported CPU vendor: {vendor}")
        except ImportError:
            raise ImportError("Please install 'py-cpuinfo' module")
    

    def _update_available_devices(self):
        """Update device selection based on chosen engine"""
        current_engine = self.EngineSelection.currentText()
        self.DeviceSelection.clear()
        available_devices = self._get_supported_devices(current_engine)
        if available_devices:
            self.DeviceSelection.addItems(available_devices)
        else:
            self._show_info(f"{current_engine} engine not available or no supported devices found")
            raise ValueError(f"{current_engine} engine not available or no supported devices found")

    def _get_supported_devices(self, engine):
        supported_devices = []
        if engine in ENGINEtoDEVICE:
            for device in ENGINEtoDEVICE[engine][self.platform]:
                if device in ["Intel CPU", "AMD CPU", "ARM CPU"]:
                    supported_devices.append(device)
                elif device == "Intel GPU" and self._check_intel_gpu():
                    supported_devices.append(device)
                elif device == "Intel NPU" and self._check_intel_npu():
                    supported_devices.append(device)
                elif device == "NVIDIA GPU" and self._check_nvidia_gpu():
                    supported_devices.append(device)
                elif device == "AMD GPU" and self._check_amd_gpu():
                    supported_devices.append(device)
                elif device == "Metal":
                    try:
                        import platform
                        if platform.system() == "Darwin":
                            supported_devices.append(device)
                    except ImportError:
                        raise ImportError("Please install 'platform' module")
                elif device == "Ascend NPU" and self._check_ascend_npu():
                        supported_devices.append(device)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        return supported_devices

    def _check_intel_gpu(self):
        """Check if Intel GPU is available in the system"""
        try:
            from openvino import Core
            core = Core()
            return 'GPU' in core.available_devices
        except:
            return False
        
    def _check_intel_npu(self):
        """Check if Intel NPU is available through OpenVINO"""
        try:
            from openvino import Core
            core = Core()
            return 'NPU' in core.available_devices
        except:
            return False
    
    def _check_amd_gpu(self):
        """Check if AMD GPU is available in the system"""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    if device.type == cl.device_type.GPU:
                        return True
            return False
        except:
            raise ImportError("Please install 'pyopencl' module")

    def _check_nvidia_gpu(self):
        """Check if NVIDIA GPU is available in the system"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cv2
                return cv2.cuda.getCudaEnabledDeviceCount() > 0
            except:
                return False

    def _check_ascend_npu(self):
        """Check if CANN environment is available for Ascend NPU"""
        try:
            from ais_bench.infer.interface import InferSession
            return True
        except ImportError:
            return False

    @Slot(bool)
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
            self._show_info("No camera detected")
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

    def _detect_available_cameras(self) -> list:
        """Detects available cameras.   
        Returns:
            list: List of available cameras with their indices as keys.
        """
        try:
            from PyCameraList.camera_device import list_video_devices
            self.camera_list ={
                device[1]: device[0] for device in list_video_devices()
            }
        except:
            self.camera_list = {}

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
        
        self._show_info(f"Selected: {os.path.basename(file_path)}")
        
        # Store the file path for later use
        self.current_video_path = file_path

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self._show_info("Failed to open video file")
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
                self._show_info("Failed to open camera")
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
        self._init_threads(processor_function=cv2.resize,
                           processor_kwargs={
                               'dsize': (self.height, self.width),
                           })
        self._start_threads()
        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Close {mode}")
    
    def _stop_preview(self):
        """Stop camera/video preview"""
        self._stop_threads()
        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Preview {mode}")
    

    def _init_threads(self, processor_function=None, processor_kwargs={}):
        source = self._get_source()
        if source is None:
            return
        if self.video_reader_thread is None:
            self.video_reader_thread = VideoReaderThread()
        if self.frame_processor_thread is None:
            self.frame_processor_thread = FrameProcessorThread()
        if self.display_thread is None:
            self.display_thread = QTimer()
        if self.Save.isChecked() and self.video_writer_thread is None:
            self.video_writer_thread = VideoWriterThread()

        # Set up video reader thread
        self.video_reader_thread.cap = cv2.VideoCapture(source)
        self.video_reader_thread.original_frame.connect(self.add_original_frame)
        self.video_reader_thread.status_update.connect(self.read_thread_status)

        # Set up frame processor thread
        self.frame_processor_thread.processing_function = processor_function
        self.frame_processor_thread.processing_kwargs = processor_kwargs
        self.frame_processor_thread.frame_processed.connect(self.add_processed_frame)
        self.frame_processor_thread.status_update.connect(self.procesed_thread_status)

        # set up video writer thread
        if self.Save.isChecked():
            self.video_writer_thread.save_path = Path.cwd() / "output.avi"
            self.video_writer_thread.fps = self.fps
            self.video_writer_thread.frame_size = (self.width, self.height)

        # Set up display qtimer thread
        self.display_thread.timeout.connect(self.display_frame)

    def _start_threads(self):
        self.video_reader_thread.start()
        self.frame_processor_thread.start()
        if self.Save.isChecked():
            self.video_writer_thread.start()
        self.display_thread.start(1000 // self.fps)

    def _stop_threads(self):
        if hasattr(self, 'video_reader_thread'):
            self.video_reader_thread.safe_stop()
            self.video_reader_thread = None
        if hasattr(self, 'frame_processor_thread'):
            self.frame_processor_thread.safe_stop()
        if hasattr(self, 'display_thread'):
            self.display_thread.stop()
            self.display_thread = None
        with self.original_frame_cache.mutex:
            self.original_frame_cache.queue.clear()
        with self.processed_frame_cache.mutex:
            self.processed_frame_cache.queue.clear()
        self.Display.clear()

    def _get_source(self):
        if self.CameraORVideos.isChecked():
            try:
                current_text = self.CameraVideosSelection.currentText()
                cam_index = self.camera_list[current_text]
                return cam_index
            except:
                self._show_info("Failed to parse camera index")
                return None
        else:
            if hasattr(self, 'current_video_path'):
                return self.current_video_path
            self._show_info("No video file selected")
            return None
    

    @Slot(np.ndarray)
    def add_original_frame(self, frame):
        """Handle original frame from camera/video source"""
        try:
            self.original_frame_cache.put(frame, block=False)
        except queue.Full:
            pass
    
    @Slot(str)
    def read_thread_status(self, message):
        #QMessageBox.information(self, "Status", message)
        self._show_info(message)
        print(message)

    @Slot(np.ndarray)
    def add_processed_frame(self, frame):
        """Handle new frame from camera/video source"""
        try:
            self.processed_frame_cache.put(frame, block=False)
        except queue.Full:
            pass

    @Slot(str)
    def procesed_thread_status(self, message):
        #QMessageBox.information(self, "Status", message)
        self._show_info(message)
        print(message)

    def display_frame(self):
        """Handle new frame from camera/video source"""

        if (self.processed_frame_cache.empty() and 
            self.video_reader_thread.isRunning() and 
            self.frame_processor_thread.isRunning()):
            return
        elif (self.processed_frame_cache.empty() and 
              not self.video_reader_thread.isRunning()):
            self._stop_threads()
            mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
            self.CheckCameraVideosConnect.setText(f"Preview {mode}")
            self.Start.setText("Start")
            self.Start.setEnabled(True)
            return
        
        frame = self.processed_frame_cache.get(block=False)

        if (self.Save.isChecked() and 
            self.video_writer_thread is not None 
            and self.video_writer_thread.isRunning()):
            self.video_writer_thread.add_frame(frame)
        
        if self.Show.isChecked():
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
        else:
            self.Display.clear()

    def onWidthSetupChanged(self, value):
        """Handle changes to the Width slider value"""
        self.width = value
        self._show_info(f"Set Camera output Width: {self.width}")
    
    def onHeightSetupChanged(self, value):
        """Handle changes to the Height slider value"""
        self.height = value
        self._show_info(f"Set Camera output Height: {self.height}")

    def onFPSSetupChanged(self, value):
        """Handle changes to the FPS slider value"""
        self.fps = value
        self._show_info(f"Set Camera output FPS: {self.fps}")
    
    def onEngineSelectionChanged(self, index):
        """Handle changes in engine selection"""
        self.engine = self.EngineSelection.itemText(index)
        self._show_info(f"Selected Inference Engine: {self.engine}")
    
    def onDeviceSelectionChanged(self, index):
        """Handle changes in device selection"""
        self.device = self.DeviceSelection.itemText(index)
        self._show_info(f"Selected Inference Device: {self.device}")
    
    def onStartClicked(self):
        """Handle when the Start button is clicked to begin processing"""
        if self.Start.text() == "Start":
            source = self._get_source()
            if source is None:
                QMessageBox.warning(self, "Warning", "Please select a source")
            # inference config
            self.inference.weights_path = self.weights_path
            self.inference.data_config = self.data_config_path
            self.inference.engine = self.engine
            self.inference.device = self.device
            self.inference.model_bits = self.ModelBits.currentText()
            self.inference.print_config()
            self.inference.model_init()
            # init threads
            self._init_threads(processor_function=self.inference.process_frame)
            # start threads
            self._start_threads()
            self.Start.setEnabled(False)
    
    def onEndClicked(self):
        """Handle when the End button is clicked to stop processing"""
        self._stop_threads()
        self.Start.setText("Start")
        self.Start.setEnabled(True)
    
    def onSaveStateChanged(self, state):
        """Handle changes to the Save output checkbox state"""
        self.inference.update_config({"save": bool(state)})
    
    def onIoUChanged(self, value):
        """Handle changes to the IOU threshold slider value"""
        self.inference.update_config({"iou": value})
    
    def onConfChanged(self, value):
        """Handle changes to the Confidence threshold slider value"""
        self.inference.update_config({"conf": value})

    def onShowStateChanged(self, state):
        """Handle changes to the Show output checkbox state"""
        self.inference.update_config({"show": bool(state)})

    def onBackgroudStateChanged(self, state):
        """Handle changes to the Show Backgroud checkbox state"""
        if state == 0:
            background = "Original"
        elif state == 1:
            background = "White"
        else:
            background = "Black"
        self.inference.update_config({"background": background})
        self.Backgroud.setText(f"{background}")
    
    def onShowClassesStateChanged(self, state):
        """Handle changes to the Show Classes checkbox state"""
        self.inference.update_config({"show_classes": bool(state)})
    
    def onShowKeypointsStateChanged(self, state):
        """Handle changes to the Show Keypoints checkbox state"""
        self.inference.update_config({"show_keypoints": bool(state)})
    
    def onShowKeypointsRadiusChanged(self, value):
        """Handle changes to the keypoint radius slider value"""
        self.inference.update_config({"radius": value})
    
    def onShowSkeletonsStateChanged(self, state):
        """Handle changes to the Show Skeletons checkbox state"""
        self.inference.update_config({"show_skeletons": bool(state)})
    
    def onShowSkeletonsLineWidthChanged(self, value):
        """Handle changes to the skeleton line width slider value"""
        self.inference.update_config({"skeleton_line_width": value})
    
    @Slot(int)
    def onShowBBoxStateChanged(self, state):
        """Handle changes to the Show Bounding Box checkbox state"""
        self.inference.update_config({"show_bbox": bool(state)})
    
    @Slot(int)
    def onShowBBoxWidthChanged(self, value):
        """Handle changes to the bounding box width slider value"""
        self.inference.update_config({"bbox_line_width": value})
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_reader_thread is not None and self.video_reader_thread.isRunning():
            self.video_reader_thread.safe_stop()
        if self.video_writer_thread is not None and self.video_writer_thread.isRunning():
            self.video_writer_thread.safe_stop()
        if self.frame_processor_thread is not None and self.frame_processor_thread.isRunning():
            self.frame_processor_thread.safe_stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = AnimalPoseInferencePage()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()