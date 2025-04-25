from PySide6.QtCore import QMetaObject, Qt, Slot, Q_ARG
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget, QMessageBox
import os
import yaml
import cv2
import numpy as np
import sys
import cpuinfo
import platform
from pathlib import Path

from .ui_animalposeinference import Ui_AnimalPoseInference
from animalposetracker.utils import VideoProcessorThread, VideoWriterThread
from animalposetracker.engine import InferenceEngine
class AnimalPoseInferencePage(QWidget, Ui_AnimalPoseInference):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)

        # constants
        self.camera_list = {}

        self.video_process_thread = VideoProcessorThread()
        self.video_writer_thread = VideoWriterThread()
        self.inference = InferenceEngine()
        self.data_config_path = None
        self.data_config = dict()
        self.weights_path = None
        

        # init button
        self.initialize_controls()
    
    def initialize_controls(self):
        """Initialize all buttons and dependent controls"""
        os.chdir(Path.cwd())

        self.tools_disabled()

        self.SelectConfigure.setEnabled(True)
        self.ShowKeypoints.setCheckState(Qt.CheckState.Checked)
        self.Show.setCheckState(Qt.CheckState.Checked)
        self.PrintInformation.setText("Please select configuration and weights")

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
    
    def _init_visualization_config(self):
        """Initialize visualization configuration"""
        self.inference.update_config(
            {
            'conf': self.Conf.value(),
            'iou': self.IoU.value(),
           'show': self.Show.isChecked(),
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
            self.PrintInformation.setText("Configuration selection cancelled")
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
            self.PrintInformation.setText(f"Configuration loaded: {filename}")
            self.SelectWeights.setEnabled(True)
            
        except yaml.YAMLError as e:
            self.PrintInformation.setText(f"Invalid YAML: {str(e)}")
            self.data_config_path = None

        except Exception as e:
            self.PrintInformation.setText(f"Error: {str(e)}")
            self.data_config_path = None

    def onSelectWeightsClicked(self):
        """Open file dialog to select model weights and 
        auto-configure supported engines and devices.

        Supported Engines:
        - OpenVINO (.onnx)
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
            "OpenVINO (select .ir or .onnx);;"
            "MMdeploy Config (select .pt, .pth or .onnx);;"
            "All Files (*)",
            options=options
        )

        if not file_path:
            self.PrintInformation.setText("Weight selection cancelled")
            return

        self.EngineSelection.setEnabled(True)
        self.DeviceSelection.setEnabled(True)
        self.CameraORVideos.setEnabled(True)
        self.CheckCameraVideosConnect.setEnabled(True)

        # Clear and prepare engine selection
        self.EngineSelection.clear()
        self.DeviceSelection.clear()
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        is_dir = file_path.is_dir()

        # Hardware detection
        device_info = self._detect_available_devices()

        # Format to engine mapping
        if file_ext == '.onnx':
            self.weights_path = file_path
            engines = []

            # Add available engines based on device support
            if device_info['intel_cpu']:
                engines.extend(["OpenCV", "OpenVINO", "Ultralytics", "MMdeploy"])
            if device_info['amd_cpu']:
                engines.extend(["OpenCV", "Ultralytics", "MMdeploy"])
            if device_info['arm_cpu']:
                engines.extend(["OpenCV", "Ultralytics", "MMdeploy"])
            if device_info['nvidia_gpu']:
                engines.extend(["OpenCV", "Ultralytics", "MMdeploy"])
            if device_info['ascend_npu']:
                engines.append("CANN")
            if device_info['hailo_npu']:
                engines.append("Hailo")
            if device_info['intel_npu']:
                engines.append("OpenVINO")

            # Remove duplicates and set UI
            engines = sorted(list(set(engines)))
            self.EngineSelection.addItems(engines)
            self.EngineSelection.setCurrentText("Ultralytics" if "Ultralytics" in engines else engines[0])
            self.PrintInformation.setText(f"ONNX model loaded ({os.path.basename(file_path)})")

        elif file_ext in ['.pt', '.pth']:
            self.weights_path = file_path
            self.EngineSelection.addItems(["Ultralytics", "MMdeploy"])
            self.PrintInformation.setText(f"PyTorch model loaded ({os.path.basename(file_path)})")

        elif file_ext == '.om':
            if device_info['ascend_npu']:
                self.weights_path = file_path
                self.EngineSelection.addItems(["CANN"])
                self.PrintInformation.setText(f"CANN model loaded ({os.path.basename(file_path)})")
            else:
                self.PrintInformation.setText("Ascend NPU not available for CANN engine")
                return

        elif file_ext == '.hef':
            if device_info['hailo_npu']:
                self.weights_path = file_path
                self.EngineSelection.addItems(["Hailo"])
                self.PrintInformation.setText(f"Hailo model loaded ({os.path.basename(file_path)})")
            else:
                self.PrintInformation.setText("Hailo NPU not detected")
                return

        elif file_ext == '.engine':
            if device_info['nvidia_gpu']:
                self.weights_path = file_path
                self.EngineSelection.addItems(["Ultralytics"])
                self.PrintInformation.setText(f"TensorRT engine loaded ({os.path.basename(file_path)})")
            else:
                self.PrintInformation.setText("NVIDIA GPU required for TensorRT")
                return

        elif is_dir and self._is_openvino_dir(file_path):
            self.weights_path = file_path
            available = []
            if device_info['intel_cpu']:
                available.append("OpenVINO")
            if device_info['nvidia_gpu']:
                available.append("OpenVINO")  # OpenVINO GPU
            available.append("Ultralytics")  # CPU fallback

            self.EngineSelection.addItems(available)
            self.EngineSelection.setCurrentText("OpenVINO" if "OpenVINO" in available else available[0])
            self.PrintInformation.setText("OpenVINO model loaded")

        else:
            self.PrintInformation.setText("Unsupported format for current platform")
            raise ValueError(f"Unsupported format: {file_ext}")

        # Update available devices based on selected engine
        self._update_available_devices()

        # Enable Start button if config is also loaded
        if hasattr(self, 'model_config_path'):
            self.Start.setEnabled(True)

    def _is_openvino_dir(self, path):
        # Placeholder implementation, you need to define the actual logic
        return True

    def _detect_available_devices(self):
        """Detect available hardware devices in the system"""
        device_info = {
            'intel_cpu': self._check_intel_cpu(),
            'amd_cpu': self._check_amd_cpu(),
            'arm_cpu': self._check_arm_cpu(),
            'nvidia_gpu': self._check_nvidia_gpu(),
            'ascend_npu': self._check_cann_environment(),
            'hailo_npu': self._check_hailo_environment(),
            'intel_npu': self._check_openvino_npu()
        }
        return device_info

    def _get_supported_devices(self, engine):
        engine_device_mapping = {
            "CANN": (["Ascend NPU"], self._check_cann_environment),
            "Hailo": (["Hailo-8"], self._check_hailo_environment),
            "OpenVINO": (["Intel CPU", "Intel NPU"], [self._check_intel_cpu, 
                                                      self._check_openvino_npu]),
            "Ultralytics": (["Intel CPU", "AMD CPU", "ARM CPU", "NVIDIA GPU"],
                            [self._check_intel_cpu, self._check_amd_cpu, 
                             self._check_arm_cpu, self._check_nvidia_gpu]),
            "OpenCV": (["Intel CPU", "AMD CPU", "ARM CPU", "NVIDIA GPU"],
                    [self._check_intel_cpu, self._check_amd_cpu, 
                     self._check_arm_cpu, self._check_nvidia_gpu]),
            "MMdeploy": (["Intel CPU", "AMD CPU", "ARM CPU", "NVIDIA GPU"],
                        [self._check_intel_cpu, self._check_amd_cpu, self._check_arm_cpu, self._check_nvidia_gpu])
        }

        if engine not in engine_device_mapping:
            raise ValueError(f"Unsupported engine: {engine}")

        devices, checks = engine_device_mapping[engine]
        if isinstance(checks, list):
            available_devices = [device for device, check in zip(devices, checks) if check()]
        else:
            if checks():
                available_devices = devices
            else:
                available_devices = []

        return available_devices

    def _update_available_devices(self):
        """Update device selection based on chosen engine"""
        current_engine = self.EngineSelection.currentText()
        self.DeviceSelection.clear()

        available_devices = self._get_supported_devices(current_engine)
        if available_devices:
            self.DeviceSelection.addItems(available_devices)
        else:
            self.PrintInformation.setText(f"{current_engine} engine not available or no supported devices found")
            raise ValueError(f"{current_engine} engine not available or no supported devices found")

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

    def _check_intel_cpu(self):
        """Check if Intel CPU is available in the system"""
        info = cpuinfo.get_cpu_info()
        brand = info.get('brand_raw', '').lower()
        return 'intel' in brand

    def _check_amd_cpu(self):
        """Check if AMD CPU is available in the system"""
        info = cpuinfo.get_cpu_info()
        brand = info.get('brand_raw', '').lower()
        return 'amd' in brand

    def _check_arm_cpu(self):
        """Check if ARM CPU is available in the system"""
        machine = platform.machine().lower()
        return 'arm' in machine or 'aarch' in machine

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
            self.PrintInformation.setText("No camera detected")
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
        
        self.PrintInformation.setText(f"Selected: {os.path.basename(file_path)}")
        
        # Store the file path for later use
        self.current_video_path = file_path

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self.PrintInformation.setText("Failed to open video file")
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
                self.PrintInformation.setText("Failed to open camera")
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
        
        source = self._get_source()
        if source is None:
            return

        self.video_process_thread.source = source
        self.video_process_thread.frame_ready.connect(self.update_frame)
        self.video_process_thread.status_update.connect(self.update_status)
        self.video_process_thread.finished.connect(self._on_finished)

        mode = "Camera" if self.CameraORVideos.isChecked() else "Video"
        self.CheckCameraVideosConnect.setText(f"Close {mode}")
        self.video_process_thread.start()

    def _stop_preview(self):
        if hasattr(self, 'video_process_thread'):
            self.video_process_thread.safe_stop()
            self.video_process_thread = None
            self.Display.clear()
    
    def _on_finished(self):
        mode = "Camera" if "Camera" in self.CameraVideosSelection.currentText() else "Video"
        self.CheckCameraVideosConnect.setText(f"Preview {mode}")

    def _get_source(self):
        if self.CameraORVideos.isChecked():
            try:
                current_text = self.CameraVideosSelection.currentText()
                cam_index = self.camera_list[current_text]
                return cam_index
            except:
                self.PrintInformation.setText("Failed to parse camera index")
                return None
        else:
            if hasattr(self, 'current_video_path'):
                return self.current_video_path
            self.PrintInformation.setText("No video file selected")
            return None

    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Handle new frame from camera/video source"""
        if (self.Save.isChecked() and 
            self.video_writer_thread is not None 
            and self.video_writer_thread.isRunning()):
            self.video_writer_thread.add_frame(frame)
            
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

    @Slot(str)
    def update_status(self, message):
        #QMessageBox.information(self, "Status", message)
        self.PrintInformation.setText(message)

    def onWidthSetupChanged(self, value):
        """Handle changes to the Width slider value"""
        self.width = value
        self.PrintInformation.setText(f"Set Camera output Width: {self.width}")
    
    def onHeightSetupChanged(self, value):
        """Handle changes to the Height slider value"""
        self.height = value
        self.PrintInformation.setText(f"Set Camera output Height: {self.height}")

    def onFPSSetupChanged(self, value):
        """Handle changes to the FPS slider value"""
        self.fps = value
        self.PrintInformation.setText(f"Set Camera output FPS: {self.fps}")
    
    def onEngineSelectionChanged(self, index):
        """Handle changes in engine selection"""
        self.engine = self.EngineSelection.itemText(index)
        self.PrintInformation.setText(f"Selected Inference Engine: {self.engine}")
    
    def onDeviceSelectionChanged(self, index):
        """Handle changes in device selection"""
        self.device = self.DeviceSelection.itemText(index)
        self.PrintInformation.setText(f"Selected Inference Device: {self.device}")
    
    def onStartClicked(self):
        """Handle when the Start button is clicked to begin processing"""
        if self.Start.text() == "Start":
            source = self._get_source()
            if source is None:
                QMessageBox.warning(self, "Warning", "Please select a source")
            # inference config
            self.inference.weights_path = self.weights_path
            self.inference.data_config = self.data_config_path
            self.inference.model_init()
            self.video_process_thread.source = source
            self.video_process_thread.processing_function = self.inference.process_frame
            if self.Save.isChecked():
                self.video_writer_thread.save_path = Path.cwd() / "output.avi"
                self.video_writer_thread.fps = self.fps
                self.video_writer_thread.frame_size = (self.width, self.height)
            self.video_process_thread.start()
            self.video_writer_thread.start()
            self.Start.setText("Pause")
        elif self.Start.text() == "Pause":
            self.Start.setText("Resume")
        else:
            self.Start.setText("Start")
    
    def onEndClicked(self):
        """Handle when the End button is clicked to stop processing"""
        pass
    
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
        if self.video_process_thread is not None and self.video_process_thread.isRunning():
            self.video_process_thread.safe_stop()
        if self.video_writer_thread is not None and self.video_writer_thread.isRunning():
            self.video_writer_thread.stop()
            self.video_writer_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = AnimalPoseInferencePage()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()