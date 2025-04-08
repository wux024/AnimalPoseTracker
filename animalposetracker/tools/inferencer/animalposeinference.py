from PySide6.QtCore import   QMetaObject, Qt, Slot, Q_ARG
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget
import os
import yaml
import cv2
import numpy as np
import sys

from .ui_animalposeinference import Ui_AnimalPoseInference

from animalposetracker.utils import PreviewThread


class AnimalPoseInference(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_AnimalPoseInference()
        self.ui.setupUi(self)

        # constants
        self.camera_list = {}

        self.preview_thread = PreviewThread()

        # init button
        self.initialize_controls()
        
        # Connect all signals to their respective slots
        self.setupConnections()
    
    def setupConnections(self):
        """Connect all UI signals to their corresponding slot functions"""
        
        # Base configuration signals
        self.ui.SelectConfigure.clicked.connect(self.onSelectConfigureClicked)
        self.ui.SelectWeights.clicked.connect(self.onSelectWeightsClicked)
        self.ui.CameraORVideos.toggled.connect(self.onCameraORVideosToggled)
        self.ui.CameraVideosSelection.currentIndexChanged.connect(self.onCameraVideosSelectionChanged)
        self.ui.CheckCameraVideosConnect.clicked.connect(self.onCheckCameraVideosConnectClicked)
        
        # Camera parameter signals
        self.ui.WidthSetup.valueChanged.connect(self.onWidthSetupChanged)
        self.ui.HeightSetup.valueChanged.connect(self.onHeightSetupChanged)
        self.ui.FPSSetup.valueChanged.connect(self.onFPSSetupChanged)
        
        # Engine and device signals
        self.ui.EngineSelection.currentIndexChanged.connect(self.onEngineSelectionChanged)
        self.ui.DeviceSelection.currentIndexChanged.connect(self.onDeviceSelectionChanged)
        
        # Start/Stop control signals
        self.ui.Start.clicked.connect(self.onStartClicked)
        self.ui.End.clicked.connect(self.onEndClicked)
        
        # Display option signals
        self.ui.Save.stateChanged.connect(self.onSaveStateChanged)
        self.ui.Show.stateChanged.connect(self.onShowStateChanged)
        self.ui.Backgroud.stateChanged.connect(self.onBackgroudStateChanged)
        self.ui.ShowClasses.stateChanged.connect(self.onShowClassesStateChanged)
        self.ui.ShowKeypoints.stateChanged.connect(self.onShowKeypointsStateChanged)
        self.ui.ShowKeypointsRadius.valueChanged.connect(self.onShowKeypointsRadiusChanged)
        self.ui.ShowSkeletons.stateChanged.connect(self.onShowSkeletonsStateChanged)
        self.ui.ShowSkeletonsLineWidth.valueChanged.connect(self.onShowSkeletonsLineWidthChanged)
        self.ui.ShowBBox.stateChanged.connect(self.onShowBBoxStateChanged)
        self.ui.ShowBBoxWidth.valueChanged.connect(self.onShowBBoxWidthChanged)
    
    def initialize_controls(self):
        """Initialize all buttons and dependent controls"""
        self.ui.SelectConfigure.setEnabled(True)
        self.ui.SelectWeights.setEnabled(False)
        self.ui.CameraORVideos.setEnabled(False)
        self.ui.CheckCameraVideosConnect.setEnabled(False)
        self.ui.WidthSetup.setEnabled(False)
        self.ui.HeightSetup.setEnabled(False)
        self.ui.FPSSetup.setEnabled(False)
        self.ui.EngineSelection.setEnabled(False)
        self.ui.DeviceSelection.setEnabled(False)
        self.ui.Start.setEnabled(False)
        self.ui.End.setEnabled(False)

        self.ui.Save.setEnabled(False)
        self.ui.Show.setEnabled(False)
        self.ui.Show.setCheckState(Qt.Checked)
        self.ui.Backgroud.setEnabled(False)
        self.ui.ShowClasses.setEnabled(False)
        self.ui.Classes.clear()
        self.ui.ShowKeypoints.setEnabled(False)
        self.ui.ShowKeypoints.setCheckState(Qt.Checked)
        self.ui.ShowKeypointsRadius.setEnabled(False)
        self.ui.ShowSkeletons.setEnabled(False)
        self.ui.ShowSkeletonsLineWidth.setEnabled(False)
        self.ui.ShowBBox.setEnabled(False)
        self.ui.ShowBBoxWidth.setEnabled(False)
        self.ui.PrintInformation.setText("Please select configuration and weights")

    def tools_enabled(self):
        self.ui.CheckCameraVideosConnect.setEnabled(True)
        self.ui.WidthSetup.setEnabled(True)
        self.ui.HeightSetup.setEnabled(True)
        self.ui.FPSSetup.setEnabled(True)
        self.ui.Start.setEnabled(True)
        self.ui.End.setEnabled(True)
        self.ui.Save.setEnabled(True)
        self.ui.Show.setEnabled(True)
        self.ui.Backgroud.setEnabled(True)
        self.ui.ShowClasses.setEnabled(True)
        self.ui.Classes.setEnabled(True)
        self.ui.ShowKeypoints.setEnabled(True)
        self.ui.ShowKeypointsRadius.setEnabled(True)
        self.ui.ShowSkeletons.setEnabled(True)
        self.ui.ShowSkeletonsLineWidth.setEnabled(True)
        self.ui.ShowBBox.setEnabled(True)

    def tools_disabled(self):
        self.ui.CheckCameraVideosConnect.setEnabled(False)
        self.ui.WidthSetup.setEnabled(False)
        self.ui.HeightSetup.setEnabled(False)
        self.ui.FPSSetup.setEnabled(False)
        self.ui.Start.setEnabled(False)
        self.ui.End.setEnabled(False)
        self.ui.Save.setEnabled(False)
        self.ui.Show.setEnabled(False)
        self.ui.Backgroud.setEnabled(False)
        self.ui.ShowClasses.setEnabled(False)
        self.ui.Classes.setEnabled(False)
        self.ui.ShowKeypoints.setEnabled(False)
        self.ui.ShowKeypointsRadius.setEnabled(False)
        self.ui.ShowSkeletons.setEnabled(False)
        self.ui.ShowSkeletonsLineWidth.setEnabled(False)
        self.ui.ShowBBox.setEnabled(False)

    
    @Slot()
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
            self.ui.PrintInformation.setText("Configuration selection cancelled")
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
            self.ui.PrintInformation.setText(f"Configuration loaded: {filename}")
            self.ui.SelectWeights.setEnabled(True)
            
        except yaml.YAMLError as e:
            self.ui.PrintInformation.setText(f"Invalid YAML: {str(e)}")
            self.data_config_path = None

        except Exception as e:
            self.ui.PrintInformation.setText(f"Error: {str(e)}")
            self.data_config_path = None

    @Slot()
    def onSelectWeightsClicked(self):
        """Open file dialog to select model weights and auto-configure supported engines and devices.
        
        Supported Engines:
        - OpenVINO (.onnx, .xml/.bin)
        - OpenCV (.onnx)
        - Ultralytics (.pt, .torchscript, .onnx, etc.)
        - MMdeploy (.onnx, .pt, deploy configs)
        - CANN (.om for Ascend NPU)
        - Hailo (.hef for Hailo NPU)
        
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
            "All Supported Formats (*.pt *.onnx *.om *.hef *.engine);;"
            "PyTorch (*.pt);;"
            "ONNX (*.onnx);;"
            "CANN Model (*.om);;"
            "Hailo Model (*.hef);;"
            "TensorRT (*.engine);;"
            "OpenVINO (select .xml or .onnx);;"
            "MMdeploy Config (select .json);;"
            "All Files (*)",
            options=options
        )
        
        if not file_path:
            self.ui.PrintInformation.setText("Weight selection cancelled")
            return
        
        self.ui.EngineSelection.setEnabled(True)
        self.ui.DeviceSelection.setEnabled(True)
        self.ui.CameraORVideos.setEnabled(True)
        self.ui.CheckCameraVideosConnect.setEnabled(True)
        
        # Clear and prepare engine selection
        self.ui.EngineSelection.clear()
        self.ui.DeviceSelection.clear()
        file_ext = os.path.splitext(file_path)[1].lower()
        is_dir = os.path.isdir(file_path)
        
        # Hardware detection
        device_info = self._detect_available_devices()
        
        # Format to engine mapping
        if file_ext == '.onnx':
            self.weights_path = file_path
            engines = []
            
            # Add available engines based on device support
            if device_info['cpu']:
                engines.extend(["OpenCV", "OpenVINO", "Ultralytics", "MMdeploy"])
            if device_info['gpu']:
                engines.extend(["OpenVINO", "Ultralytics"])  # GPU-specific engines
            if device_info['ascend_npu']:
                engines.append("CANN")
                
            # Remove duplicates and set UI
            engines = sorted(list(set(engines)))
            self.ui.EngineSelection.addItems(engines)
            self.ui.EngineSelection.setCurrentText("OpenVINO" if "OpenVINO" in engines else engines[0])
            self.ui.PrintInformation.setText(f"ONNX model loaded ({os.path.basename(file_path)})")
            
        elif file_ext == '.pt':
            self.weights_path = file_path
            self.ui.EngineSelection.addItems(["Ultralytics", "MMdeploy"])
            self.ui.PrintInformation.setText(f"PyTorch model loaded ({os.path.basename(file_path)})")
            
        elif file_ext == '.om':
            if device_info['ascend_npu']:
                self.weights_path = file_path
                self.ui.EngineSelection.addItems(["CANN"])
                self.ui.PrintInformation.setText(f"CANN model loaded ({os.path.basename(file_path)})")
            else:
                self.ui.PrintInformation.setText("Ascend NPU not available for CANN engine")
                return
                
        elif file_ext == '.hef':
            if device_info['hailo_npu']:
                self.weights_path = file_path
                self.ui.EngineSelection.addItems(["Hailo"])
                self.ui.PrintInformation.setText(f"Hailo model loaded ({os.path.basename(file_path)})")
            else:
                self.ui.PrintInformation.setText("Hailo NPU not detected")
                return
                
        elif file_ext == '.engine':
            if device_info['gpu']:
                self.weights_path = file_path
                self.ui.EngineSelection.addItems(["Ultralytics"])
                self.ui.PrintInformation.setText(f"TensorRT engine loaded ({os.path.basename(file_path)})")
            else:
                self.ui.PrintInformation.setText("NVIDIA GPU required for TensorRT")
                return
                
        elif is_dir and self._is_openvino_dir(file_path):
            self.weights_path = file_path
            available = []
            if device_info['cpu']: available.append("OpenVINO")
            if device_info['gpu']: available.append("OpenVINO")  # OpenVINO GPU
            available.append("Ultralytics")  # CPU fallback
            
            self.ui.EngineSelection.addItems(available)
            self.ui.EngineSelection.setCurrentText("OpenVINO" if "OpenVINO" in available else available[0])
            self.ui.PrintInformation.setText("OpenVINO model loaded")
            
        elif file_ext == '.json' and self._is_mmdeploy_config(file_path):
            self.weights_path = file_path
            self.ui.EngineSelection.addItems(["MMdeploy"])
            self.ui.PrintInformation.setText("MMdeploy config loaded")
            
        else:
            self.ui.PrintInformation.setText("Unsupported format for current platform")
            return
        
        # Update available devices based on selected engine
        self._update_available_devices()
        
        # Enable Start button if config is also loaded
        if hasattr(self, 'model_config_path'):
            self.ui.Start.setEnabled(True)

    def _detect_available_devices(self):
        """Detect available hardware devices in the system"""
        device_info = {
            'cpu': True,  # Always available
            'gpu': self._check_nvidia_gpu(),
            'ascend_npu': self._check_cann_environment(),
            'hailo_npu': self._check_hailo_environment(),
            'intel_npu': self._check_openvino_npu()
        }
        return device_info

    def _update_available_devices(self):
        """Update device selection based on chosen engine"""
        current_engine = self.ui.EngineSelection.currentText()
        self.ui.DeviceSelection.clear()
        
        if current_engine == "CANN":
            self.ui.DeviceSelection.addItems(["Ascend NPU"])
        elif current_engine == "Hailo":
            self.ui.DeviceSelection.addItems(["Hailo-8"])
        elif current_engine == "OpenVINO":
            devices = ["CPU"]
            if self._check_nvidia_gpu(): devices.append("GPU")
            if self._check_openvino_npu(): devices.append("Intel NPU")
            self.ui.DeviceSelection.addItems(devices)
        else:  # General engines
            devices = ["CPU"]
            if self._check_nvidia_gpu(): devices.append("NVIDIA GPU")
            self.ui.DeviceSelection.addItems(devices)

    def _check_nvidia_gpu(self):
        """Check if NVIDIA GPU is available in the system"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return tf.test.is_gpu_available()
            except:
                # Fallback to OpenCV check
                try:
                    import cv2
                    return cv2.cuda.getCudaEnabledDeviceCount() > 0
                except:
                    return False

    def _check_cann_environment(self):
        """Check if CANN environment is available for Ascend NPU"""
        try:
            from ais_bench.infer.interface import InferSession
            return True
        except ImportError:
            return False

    def _check_hailo_environment(self):
        """Check if Hailo runtime is available"""
        try:
            import hailort
            return True
        except ImportError:
            return False

    def _check_openvino_npu(self):
        """Check if Intel NPU is available through OpenVINO"""
        try:
            from openvino.runtime import Core
            core = Core()
            return 'NPU' in core.available_devices
        except:
            return False

    def _detect_available_devices(self):
        """Detect available hardware devices in the system"""
        return {
            'cpu': True,  # Always available
            'gpu': self._check_nvidia_gpu(),
            'ascend_npu': self._check_cann_environment(),
            'hailo_npu': self._check_hailo_environment(),
            'intel_npu': self._check_openvino_npu()
        }
    
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
        self.ui.WidthSetup.setRange(1, 9999)
        self.ui.HeightSetup.setRange(1, 9999)
        self.ui.FPSSetup.setRange(1, 500)
        self.ui.WidthSetup.setValue(width if width > 0 else 640)
        self.ui.HeightSetup.setValue(height if height > 0 else 480)
        self.ui.FPSSetup.setValue(fps if fps > 0 else 30)

    def _setup_camera_ui(self):
        """Initialize camera mode UI according to the original design"""
        # Set camera mode toggle state
        self.ui.CameraORVideos.setChecked(True)
        self.ui.CameraORVideos.setText("Use Video")  # Ensure text matches design
        self.ui.CheckCameraVideosConnect.setText("Preview Camera")
        
        # Configure camera selection dropdown
        self.ui.CameraVideosSelection.clear()
        self._detect_available_cameras()
        if not self.camera_list:
            self.ui.PrintInformation.setText("No camera detected")
        else:
            self.ui.CameraVideosSelection.addItem("Select camera...")
            self.ui.CameraVideosSelection.addItems(self.camera_list.keys())
        
        # Set default camera parameters (from design)
        self._set_camera_setup()

    def _setup_video_ui(self):
        """Initialize file mode UI according to the original design"""
        # Set file mode toggle state
        self.ui.CameraORVideos.setChecked(False)
        self.ui.CameraORVideos.setText("Use Camera")
        
        # Configure file selection
        self.ui.CameraVideosSelection.clear()
        self.ui.CameraVideosSelection.addItem("...")
        self.ui.CameraVideosSelection.addItem("Select video file...")

        self.ui.CheckCameraVideosConnect.setText("Preview Video")

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
        
        self.ui.PrintInformation.setText(f"Selected: {os.path.basename(file_path)}")
        
        # Store the file path for later use
        self.current_video_path = file_path

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self.ui.PrintInformation.setText("Failed to open video file")
            return
        
        # Update UI with actual video parameters
        self._set_camera_setup(cap)
        cap.release()

    def _update_camera_params(self):
        """Read and display parameters from selected camera"""
        try:
            current_text = self.ui.CameraVideosSelection.currentText()
            cam_index = self.camera_list[current_text]
            cap = cv2.VideoCapture(cam_index)
            
            if not cap.isOpened():
                self.ui.PrintInformation.setText("Failed to open camera")
                return
            
            self.tools_enabled()
            self._set_camera_setup(cap)
            cap.release()
            
        except Exception as e:
            print(f"Error updating camera params: {str(e)}")
            # Fallback to defaults
            self._set_camera_setup()
        
    
    @Slot(int)
    def onCameraVideosSelectionChanged(self):
        """Handle changes in camera/video source selection"""
        if self.ui.CameraVideosSelection.currentText() in self.camera_list.keys():
            self._update_camera_params()
        elif "Select video file..." in self.ui.CameraVideosSelection.currentText():
            self._select_video_file()
            self.ui.CameraVideosSelection.setCurrentText("...")
        else:
            pass
    
    @Slot()
    def onCheckCameraVideosConnectClicked(self):
        """Handle when checking camera connection is requested"""

        btn = self.ui.CheckCameraVideosConnect
        current_text = btn.text()
        btn.setEnabled(False)
        try:
            if current_text == "Preview Camera":
                self._start_preview(is_camera=True)
            elif current_text == "Close Camera":
                self._stop_preview()
            elif current_text == "Preview Video":
                self._start_preview(is_camera=False)
            elif current_text == "Close Video":
                self._stop_preview()
        finally:
            btn.setEnabled(True)
    
    def _start_preview(self, is_camera: bool):
        if hasattr(self, 'preview_thread'):
            self._stop_preview()

        source = self._get_preview_source(is_camera)
        if source is None:
            return

        self.preview_thread = PreviewThread(source=source)
        self.preview_thread.frame_ready.connect(self.update_preview_frame)
        self.preview_thread.status_update.connect(self.update_preview_status)
        self.preview_thread.finished.connect(self._on_preview_finished)

        mode = "Camera" if is_camera else "Video"
        self.ui.CheckCameraVideosConnect.setText(f"Close {mode}")
        self.preview_thread.start()

    def _stop_preview(self):
        if hasattr(self, 'preview_thread'):
            self.preview_thread.safe_stop()
            self.previe_thread = None
    
    def _on_preview_finished(self):
        mode = "Camera" if "Camera" in self.ui.CameraVideosSelection.currentText() else "Video"
        self.ui.CheckCameraVideosConnect.setText(f"Preview {mode}")
        self.ui.Display.setPixmap(QPixmap(u"assets/logo.png"))

    def _get_preview_source(self, is_camera):
        if is_camera:
            try:
                current_text = self.ui.CameraVideosSelection.currentText()
                cam_index = self.camera_list[current_text]
                return cam_index
            except:
                self.ui.PrintInformation.setText("Failed to parse camera index")
                return None
        else:
            if hasattr(self, 'current_video_path'):
                return self.current_video_path
            self.ui.PrintInformation.setText("No video file selected")
            return None

    @Slot(np.ndarray)
    def update_preview_frame(self, frame):
        """Handle new frame from camera/video source"""
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = RGB.shape
        QImg = QImage(RGB.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg).scaled(
            self.ui.Display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        QMetaObject.invokeMethod(
            self.ui.Display,
            "setPixmap",
            Qt.QueuedConnection,
            Q_ARG(QPixmap, pixmap)
        )

    @Slot(str)
    def update_preview_status(self, message):
        self.ui.PrintInformation.setText(message)

    @Slot(int)
    def onWidthSetupChanged(self):
        """Handle changes to the width parameter setting"""
        self.width = self.ui.WidthSetup.value()
        self.ui.PrintInformation.setText(f"Set Camera output Width: {self.width}")
    
    @Slot(int)
    def onHeightSetupChanged(self):
        """Handle changes to the height parameter setting"""
        self.height = self.ui.HeightSetup.value()
        self.ui.PrintInformation.setText(f"Set Camera output Height: {self.height}")

    @Slot(int)
    def onFPSSetupChanged(self):
        """Handle changes to the FPS parameter setting"""
        self.fps = self.ui.FPSSetup.value()
        self.ui.PrintInformation.setText(f"Set Camera output FPS: {self.fps}")
    
    @Slot(int)
    def onEngineSelectionChanged(self):
        """Handle changes to the inference engine selection"""
        self.engine = self.ui.EngineSelection.currentText()
        self.ui.PrintInformation.setText(f"Selected Inference Engine: {self.engine}")
    
    @Slot(int)
    def onDeviceSelectionChanged(self):
        """Handle changes to the inference device selection"""
        self.device = self.ui.DeviceSelection.currentText()
        self.ui.PrintInformation.setText(f"Selected Inference Device: {self.device}")
    
    @Slot()
    def onStartClicked(self):
        """Handle when the Start button is clicked to begin processing"""
        if self.ui.Start.text() == "Start":
            self.ui.Start.setText("Pause")
        elif self.ui.Start.text() == "Pause":
            self.ui.Start.setText("Resume")
        else:
            self.ui.Start.setText("Start")
    
    @Slot()
    def onEndClicked(self):
        """Handle when the End button is clicked to stop processing"""
        pass
    
    @Slot(int)
    def onSaveStateChanged(self, state):
        """Handle changes to the Save output checkbox state"""
        self.save = bool(state)

    @Slot(int)
    def onShowStateChanged(self, state):
        """Handle changes to the Show output checkbox state"""
        self.show_ = bool(state)

    @Slot(int)
    def onBackgroudStateChanged(self, state):
        """Handle changes to the Show Backgroud checkbox state"""
        if state == 0:
            self.ui.Backgroud.setText("Original")
        elif state == 1:
            self.ui.Backgroud.setText("White")
        else:
            self.ui.Backgroud.setText("Black")
        self.show_Backgroud = state
    
    @Slot(int)
    def onShowClassesStateChanged(self, state):
        """Handle changes to the Show Classes checkbox state"""
        self.show_classes = bool(state)
    
    @Slot(int)
    def onShowKeypointsStateChanged(self, state):
        """Handle changes to the Show Keypoints checkbox state"""
        self.show_keypoints = bool(state)
    
    @Slot(int)
    def onShowKeypointsRadiusChanged(self, value):
        """Handle changes to the keypoint radius slider value"""
        self.kpt_radius = value
    
    @Slot(int)
    def onShowSkeletonsStateChanged(self, state):
        """Handle changes to the Show Skeletons checkbox state"""
        self.show_skeletons = bool(state)
    
    @Slot(int)
    def onShowSkeletonsLineWidthChanged(self, value):
        """Handle changes to the skeleton line width slider value"""
        self.line_width = value
    
    @Slot(int)
    def onShowBBoxStateChanged(self, state):
        """Handle changes to the Show Bounding Box checkbox state"""
        self.show_bbox = bool(state)
    
    @Slot(int)
    def onShowBBoxWidthChanged(self, value):
        """Handle changes to the bounding box width slider value"""
        self.bbox_width = value
    
    def closeEvent(self, event):
        if self.preview_thread.isRunning():
            self.preview_thread.safe_stop()
        event.accept()

def run():
    platform = sys.platform
    os.environ["QT_QPA_PLATFORM"] = "offscreen" if platform == "linux" else "windows"
    app = QApplication(sys.argv)
    window = AnimalPoseInference()
    window.show()
    sys.exit(app.exec())