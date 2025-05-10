from PySide6.QtCore import QMetaObject, Qt, Q_ARG, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget, QMessageBox
import os
import yaml
import cv2
import sys
from pathlib import Path
import queue

from .ui_inferencer import Ui_Inferencer
from animalposetracker.utils import (VideoReaderThread, VideoWriterThread, 
                                     PreprocessThread, InferenceThread, 
                                     PostprocessThread, VisualizeThread)
from animalposetracker.engine import InferenceEngine
from animalposetracker.engine import ENGINEtoDEVICE
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
        self.max_cache_size = 1024

        self.inference = None
        self.device = None
        self.engine = None

        # threads
        self.video_reader_thread = None
        self.video_writer_thread = None
        self.preprocess_thread = None
        self.inference_thread = None
        self.postprocess_thread = None
        self.visualize_thread = None

        self.data_config_path = None
        self.weights_path = None

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

        self._enable_widgets()
        self._clear_selections()
        self.supported_engines_and_devices = {}

        self.weights_path = file_path
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        

        self._get_supported_engines_and_devices(file_ext)

        if not self.supported_engines_and_devices:
            raise ValueError(f"Unsupported format: {file_ext}")

        self._init_engine_selection(list(self.supported_engines_and_devices.keys()))
        self._init_available_devices()
        self._init_model_bits()

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
            Warning("No camera detected")
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
            Warning("Failed to open video file")
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
                Warning("Failed to open camera")
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
            Warning("Failed to open camera/video source")
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
        return {
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
            Warning("Failed to open camera/video source")
        if self.CameraORVideos.isChecked():
            self._set_camera_params(cap)
        self.videoreader_thread.cap = cap
        self.videoreader_thread.status_update.connect(self.thread_status)
        self.videoreader_thread.data_ready.connect(self.put_read_data)
        self.videoreader_thread.finished.connect(self.read_end)

        # Set up frame preprocessor thread
        self.preprocess_thread.preprocess_function = self.inference.preprocess
        self.preprocess_thread.status_update.connect(self.thread_status)
        self.preprocess_thread.data_ready.connect(self.put_preprocessed_data)

        # Set up inference thread
        self.inference_thread.inference_function = self.inference.inference
        self.inference_thread.status_update.connect(self.thread_status)
        self.inference_thread.data_ready.connect(self.put_inference_data)

        # Set up postprocessor thread
        self.postprocess_thread.postprocess_function = self.inference.postprocess
        self.postprocess_thread.status_update.connect(self.thread_status)
        self.postprocess_thread.data_ready.connect(self.put_postprocessed_data)

        # Set up visualization thread
        self.visualize_thread.visualize_function = self.inference.visualize
        self.visualize_thread.data_ready.connect(self.put_visualized_data)
        self.visualize_thread.status_update.connect(self.thread_status)

        # Set up display thread
        self.display_thread.timeout.connect(self.display_inferece_frame)

        # set up video writer thread
        if self.Save.isChecked():
            if self.Tracker.isChecked():
                save_path = Path.home() / "runs" / "inference"
            save_path.mkdir(exist_ok=True, parents=True)
            self.videowriter_thread.save_path = save_path / "output.avi"
            self.videowriter_thread.fps = self.fps
            self.videowriter_thread.frame_size = (self.width, self.height)


    def _start_inference_threads(self):
        self.videoreader_thread.start()
        self.preprocess_thread.start()
        self.inference_thread.start()
        self.postprocess_thread.start()
        self.visualize_thread.start()
        self.display_thread.start(1000 // self.fps)
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
                Warning("Failed to parse camera index")
                return None
        else:
            if hasattr(self, 'current_video_path'):
                return self.current_video_path
            Warning("No video file selected")
            return None
    
    def thread_status(self, message):
        print(message)
    
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
            self._stop_all_threads()
            return
        
        frame, results = self.visualize_cache.get()

        if (self.Save.isChecked() and 
            self.videowriter_thread is not None 
            and self.videowriter_thread.isRunning()):
            self.videowriter_thread.add_frame(frame, results)
        
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
            # inference config
            self.inference = InferenceEngine()
            self.inference.weights_path = self.weights_path
            self.inference.data_config = self.data_config_path
            self.inference.engine = self.engine
            self.inference.device = self.device
            self.inference.model_bits = self.model_bits
            self.inference.update_config(self._init_visualization_config())
            self.inference.model_init()
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
            self.Display.clear()
            self.Start.setText("Start")
            self.Start.setEnabled(True)
        finally:
            if hasattr(self, '_stopping'):
                del self._stopping

    
    def onIoUChanged(self, value):
        """Handle changes to the IOU threshold slider value"""
        self.inference.update_config({"iou": value})
    
    def onConfChanged(self, value):
        """Handle changes to the Confidence threshold slider value"""
        self.inference.update_config({"conf": value})

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
    
    def onShowBBoxStateChanged(self, state):
        """Handle changes to the Show Bounding Box checkbox state"""
        self.inference.update_config({"show_bbox": bool(state)})
    
    def onShowBBoxWidthChanged(self, value):
        """Handle changes to the bounding box width slider value"""
        self.inference.update_config({"bbox_line_width": value})

    def onFPSShowStateChanged(self, state):
        """Handle changes to the Show FPS checkbox state"""
        self.inference.update_config({"show_fps": bool(state)})
    
    def onPreprocessTimeStateChanged(self, state):
        """Handle changes to the Show Preprocess Time checkbox state"""
        self.inference.update_config({"show_preprocess_time": bool(state)})
    
    def onInferenceTimeStateChanged(self, state):
        """Handle changes to the Show Inference Time checkbox state"""
        self.inference.update_config({"show_inference_time": bool(state)})
    
    def onPostprocessTimeStateChanged(self, state):
        """Handle changes to the Show Postprocess Time checkbox state"""
        self.inference.update_config({"show_postprocess_time": bool(state)})
    
    def onFontScaleChanged(self, value):
        """Handle changes to the Font Scale slider value"""
        self.inference.update_config({"font_scale": value / 10.0})
    
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