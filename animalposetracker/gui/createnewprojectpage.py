from PySide6.QtCore import Signal, QTimer, Qt, QSignalBlocker
from PySide6.QtWidgets import  (QFileDialog, QWidget, QMenu)
from PySide6.QtGui import QCursor
from pathlib import Path

from .ui_createnewproject import Ui_CreateNewProject
from .utils import ListManager, CheckableListWidgetItem
from animalposetracker.projector import AnimalPoseTrackerProject
from animalposetracker.cfg import MODEL_YAML_PATHS

class CreateNewProjectPage(QWidget, Ui_CreateNewProject):
    # Signals
    CreateProjectClicked = Signal()
    source_type = "image"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        self.initialize_controls()

    
    def initialize_controls(self):
        """Initialize all UI controls based on project configuration"""

        self.project = AnimalPoseTrackerProject()
        self.project.local_path = Path.cwd()

        # Load project config
        self.SourceDataSelected.setTristate(True)
        self.SourceDataSelected.setEnabled(True)

        self.LocationPathDisplay.setReadOnly(True)
        self.LocationPathDisplay.setText(str(self.project.local_path))

        # edit project timer
        self.edit_project_timer = QTimer()
        self.edit_project_timer.setSingleShot(True)
        self.edit_project_timer.timeout.connect(self.onEditProjectTimerTimeout)

        self.worker_project_timer = QTimer()
        self.worker_project_timer.setSingleShot(True)
        self.worker_project_timer.timeout.connect(self.onWorkerProjectTimerTimeout)

        self.ModelTypeSelection.clear()
        self.ModelTypeSelection.addItems(list(MODEL_YAML_PATHS.keys()))

        self.keypoint_manager =  ListManager(
                    scroll_area=self.KeypointList,
                    initial_widgets={
                        'label': self.KeypointNumber1,
                        'input': self.Keypoint1,
                        'h_layout': self.KeypointLayout,
                        'v_layout': self.KeypointListVLayout,
                        'container': self.KeypointListLayout
                    })
        self.keypoint_manager.list_changed.connect(self.onKeypointChanged)

        self.class_manager =  ListManager(
                    scroll_area=self.ClassList,
                    initial_widgets={
                        'label': self.ClassNumber1,
                        'input': self.Class1,
                        'h_layout': self.ClassLayout,
                        'v_layout': self.ClassListVLayout,
                        'container': self.ClassListLayout
                    })
        self.class_manager.list_changed.connect(self.onClassChanged)



    def setupConnections(self):
        """Initialize all signal-slot connections for this UI"""

        self.ProjectConfig.textEdited.connect(self.onProjectTextEdited)
        self.WorkerConfig.textEdited.connect(self.onWorkerTextEdited)

        # Path selection
        self.LocationPathSelection.clicked.connect(self.onLocationPathClicked)
        
        # Source data operations
        self.BrowseSourceData.clicked.connect(self.onBrowseSourceDataClicked)
        self.ClearSourceDataList.clicked.connect(self.onClearSourceDataClicked)
        
        # Project creation
        self.CreateProjectBase.clicked.connect(self.onCreateProjectClicked)
        
        # Checkbox states
        self.SourceDataSelected.stateChanged.connect(self.onSourceDataSelectionChanged)
        
        # Configuration changes
        self.KeypointConfig.valueChanged.connect(self.onKeypointConfigChanged)
        self.ClassConfig.valueChanged.connect(self.onClassConfigChanged)
        
        # Model selection
        self.ModelTypeSelection.currentTextChanged.connect(self.onModelTypeChanged)
        self.ModelScaleSelection.currentTextChanged.connect(self.onModelScaleChanged)

    def onProjectTextEdited(self):
        """Handler for project configuration text edit"""
        self.edit_project_timer.stop()
        self.edit_project_timer.start(500) # wait for 500ms before updating project config
        
    def onEditProjectTimerTimeout(self):
        """Handler for edit project timer timeout"""
        self.project.update_config("project", {"project_name": self.ProjectConfig.text()})

    def onWorkerTextEdited(self):
        """Handler for worker configuration text edit"""
        self.worker_project_timer.stop()
        self.worker_project_timer.start(500) # wait for 500ms before updating worker config
        
    def onWorkerProjectTimerTimeout(self):
        """Handler for worker project timer timeout"""
        self.project.update_config("project", {"worker": self.WorkerConfig.text()})
    
    def onLocationPathClicked(self):
        """Open directory dialog to select project location and update UI"""
        start_dir = str(Path.cwd())
        selected_dir = QFileDialog.getExistingDirectory(self, 
                                                        "Select Directory", 
                                                        start_dir,
                                                         QFileDialog.ShowDirsOnly)
        if selected_dir:
            self.project.local_path = Path(selected_dir)
            self.LocationPathDisplay.setText(str(self.project.local_path))
        
    def onBrowseSourceDataClicked(self):
        """
        Handles media source selection with context menu for type selection.
        Maintains single button interaction with intelligent type detection.
        """
        # Create context menu
        menu = QMenu(self)
        video_action = menu.addAction("Select Video(s)")
        image_action = menu.addAction("Select Image Directory")
        
        # Show menu at cursor position
        action = menu.exec_(QCursor.pos())
        
        if action == video_action:
            self.source_type = "video"
            self._handleVideoSelection()
        elif action == image_action:
            self.source_type = "image"
            self._handleImageDirSelection()

    def _handleVideoSelection(self):
        """Process video files/directory selection"""
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
        
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Video(s)")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        
        if dialog.exec():
            selected = dialog.selectedFiles()
            valid = []
            
            # Process mixed selection (files + directories)
            for path in selected:
                path_obj = Path(path)
                if path_obj.is_dir():
                    # Scan directory for videos
                    dir_videos = [str(f) for f in path_obj.glob('*') 
                                if f.suffix.lower() in VIDEO_EXTS]
                    valid.extend(sorted(dir_videos))
                else:
                    if path_obj.suffix.lower() in VIDEO_EXTS:
                        valid.append(str(path_obj))
            
            if valid:
                self._updateSourceList(valid, len(valid))
            else:
                print("No Valid Videos", "Selected paths contain no video files")

    def _handleImageDirSelection(self):
        """Process image directory selection with strict validation"""
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Image Directory")
        dialog.setFileMode(QFileDialog.Directory)
        
        if dialog.exec():
            dir_path = Path(dialog.selectedFiles()[0])
            all_files = list(dir_path.glob('*'))
            
            images = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]
            non_images = [f for f in all_files if f.suffix.lower() not in IMAGE_EXTS]
            
            if len(images) == 0:
                self.showErrorMessage("Empty Directory", "No images found in selected folder")
            elif non_images:
                self.showErrorMessage("Invalid Directory", 
                                    f"Contains {len(non_images)} non-image files")
            else:
                self._updateSourceList([f"{dir_path}"], len(images))

    def _updateSourceList(self, items, count):
        """Update UI with validation markers"""
        self.SourceList.clear()
        # Add type indicators to source list
        with QSignalBlocker(self.SourceDataSelected):
            for item in items:
                item = CheckableListWidgetItem(str(item))
                self.SourceList.addItem(item)
                self.SourceList.setItemWidget(item, item.checkbox)
                item.checked_signal.source_selected.connect(self._handleSourceDataSelection)

        self.SourceDataSelected.setCheckState(Qt.CheckState.Checked)
        self.SourceDataSelected.setText(f"{count} {self.source_type}{'s' if count > 1 else ''} selected")
    
    def _handleSourceDataSelection(self):
        checked_items = self.GetCheckedItems()
        self._updateSourceDataSelectionChanged(len(checked_items))
        self.project.update_config("project", {"sources": checked_items})
    
    def _updateSourceDataSelectionChanged(self, checked_count):
        """Update UI with source data selection summary"""
        with QSignalBlocker(self.SourceDataSelected):
            if checked_count == 0:
                state = Qt.Unchecked
            elif checked_count == self.SourceList.count():
                state = Qt.Checked
            else:
                state = Qt.PartiallyChecked
            self.SourceDataSelected.setCheckState(state)
            self.SourceDataSelected.setText(
                f"{checked_count} {self.source_type}{'s' if checked_count > 1 else ''} selected"
            )
    
    def onSourceDataSelectionChanged(self):
        """Handler for source data selection state change"""
        state = self.SourceDataSelected.checkState()
        if state == Qt.CheckState.PartiallyChecked:
            return
        with QSignalBlocker(self.SourceDataSelected):
            for i in range(self.SourceList.count()):
                item = self.SourceList.item(i)
                if not item:
                    continue
                if hasattr(item, 'checkbox'):
                    item.checkbox.setCheckState(state)
        checked_items = self.GetCheckedItems()
        self._updateSourceDataSelectionChanged(len(checked_items))
        self.project.update_config("project", {"sources": checked_items})

    def GetCheckedItems(self):
        checked = []
        for i in range(self.SourceList.count()):
            item = self.SourceList.item(i)
            if not item:
                continue
            if hasattr(item, 'checkbox') and item.checkbox.isChecked():
                checked.append(item.checkbox.text())
        return checked

    def onClearSourceDataClicked(self):
        """Handler for clearing source data list"""
        self.SourceList.clear()
        self.SourceDataSelected.setCheckState(Qt.CheckState.Unchecked)
        self.SourceDataSelected.setText("0 source data selected")
        self.project.update_config("project", {"sources": []})
        

    def getAllProjectConfig(self):
        project_config = {
            "project_path": str(self.project.project_path),
            "project_name": self.ProjectConfig.text(),
            "worker": self.WorkerConfig.text(),
            "model_type": self.ModelTypeSelection.currentText(),
            "model_scale": self.ModelScaleSelection.currentText(),
            "keypoints": self.KeypointConfig.value(),
            "visible": self.VisibleSelection.isChecked(),
            "keypoints_name": self.keypoint_manager.get_list_data(),
            "classes": self.ClassConfig.value(),
            "classes_name": self.class_manager.get_list_data(),
            "source": self.GetCheckedItems(),
        }
        if not project_config["keypoints_name"]:
            project_config["keypoints_name"] = [f"keypoint{i+1}" for i in range(project_config["keypoints"])]
        if not project_config["classes_name"]:
            project_config["classes_name"] = [f"class{i+1}" for i in range(project_config["classes"])]

        return project_config

    def onCreateProjectClicked(self):
        project_config = self.getAllProjectConfig() 
        if project_config is None:
            return
        self.project.update_config("project", project_config)
        self.project.local_path = Path(self.LocationPathDisplay.text())
        self.project.create_new_project()

        if self.CopySourceData.isChecked():
            sources = self.project.project_config['sources']
            self.project.add_source_to_project(sources, 'copy')

        self.CreateProjectClicked.emit()
        
    def onKeypointConfigChanged(self, keypoints):
        self.project.update_config("project", {"keypoints": keypoints})
        
    def onClassConfigChanged(self, classes):
        self.project.update_config("project", {"classes": classes})
        
    def onModelTypeChanged(self, model_type):
        """Handler for model type selection change"""
        if model_type == "AnimalViTPose":
            self.ModelScaleSelection.clear()
            self.ModelScaleSelection.addItems(["S", "B", "L", "H"])
        else:
            self.ModelScaleSelection.clear()
            self.ModelScaleSelection.addItems(["N", "S", "M", "L", "X"])
        self.project.update_config("model", {"model_type": model_type})
        
    def onModelScaleChanged(self, model_scale):
        """Handler for model scale selection change"""
        self.project.update_config("model", {"model_scale": model_scale})

    def onKeypointChanged(self):
        """Handler for keypoint input change"""
        keypoint_names = self.keypoint_manager.get_list_data()
        self.project.update_config("project", {"keypoints_names": keypoint_names})

    def onClassChanged(self):
        """Handler for class input change"""
        class_names = self.class_manager.get_list_data()
        self.project.update_config("project", {"classes_name": class_names})
    

        
        