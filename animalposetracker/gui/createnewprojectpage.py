from PySide6.QtCore import Signal, QTimer, Qt, QSize, QObject
from PySide6.QtWidgets import  (QFileDialog, QWidget, QHBoxLayout, 
    QVBoxLayout, QLabel, QLineEdit, QLayout)
import os
import yaml
import cv2
import numpy as np
import sys
from pathlib import Path
import shutil

from .ui_createnewproject import Ui_CreateNewProject
from animalposetracker.projector import AnimalPoseTrackerProject
from animalposetracker.cfg import MODEL_YAML_PATHS

class ListManager(QObject):
    list_changed = Signal()

    def __init__(self, scroll_area, initial_widgets):
        super().__init__()
        self.scroll_area = scroll_area
        self.container = initial_widgets['container']
        self.v_layout = initial_widgets['v_layout']
        
        # Store all input rows while preserving your exact layout structure
        self.input_rows = [{
            'number_label': initial_widgets['label'],
            'input_field': initial_widgets['input'],
            'row_layout': initial_widgets['h_layout']
        }]
        
        # Configure first row to match your screenshot exactly
        initial_widgets['label'].setText("1.")
        initial_widgets['label'].setAlignment(Qt.AlignCenter)
        initial_widgets['input'].setStyleSheet("""
            QLineEdit {
                border: none;
                border-bottom: 1px solid #ccc;
                padding: 2px 0;
            }
            QLineEdit:focus {
                border-bottom: 1px solid #1E90FF;
            }
        """)
        
        # Connect signals while maintaining your layout constraints
        initial_widgets['input'].textChanged.connect(self._handle_input_change)

    def ensure_single_empty_row(self):
        """Ensures that there is at most one empty row in the list"""
        while len(self.input_rows) > 1:
            last_row = self.input_rows[-1]
            if last_row['input_field'].text().strip():
                break
            self._remove_last_row()
        
        if not self.input_rows[-1]['input_field'].text().strip():
            return  
            
        self._add_input_row()
        self.input_rows[-1]['input_field'].clear()

    def _handle_input_change(self):
        """Adds new row when last input has content, matching your layout style"""
        current_text = self.input_rows[-1]['input_field'].text().strip()
        if current_text and len(self.input_rows) == self.v_layout.count():
            self._add_input_row()
        self.ensure_single_empty_row()
        self.list_changed.emit()

    def _add_input_row(self):
        """Creates new row while preserving your exact layout parameters"""
        row_number = len(self.input_rows) + 1
        
        # Create new layout with your exact specifications
        row_layout = QHBoxLayout()
        row_layout.setSpacing(10)
        row_layout.setContentsMargins(10, 10, -1, -1)
        row_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        
        # Create label matching your original properties
        number_label = QLabel(self.container)
        number_label.setObjectName(f"KeypointNumber{row_number}")
        number_label.setSizePolicy(self.input_rows[0]['number_label'].sizePolicy())
        number_label.setMinimumSize(QSize(10, 25))
        number_label.setMaximumSize(QSize(16777215, 25))
        number_label.setAlignment(Qt.AlignCenter)
        number_label.setText(f"{row_number}.")
        
        # Create input field matching your original properties
        input_field = QLineEdit(self.container)
        input_field.setSizePolicy(self.input_rows[0]['input_field'].sizePolicy())
        input_field.setMinimumSize(QSize(400, 25))
        input_field.setMaximumSize(QSize(16777215, 25))
        input_field.setStyleSheet(self.input_rows[0]['input_field'].styleSheet())
        input_field.textChanged.connect(self._handle_input_change)
        
        # Add to layout while maintaining your structure
        row_layout.addWidget(number_label)
        row_layout.addWidget(input_field)
        self.v_layout.addLayout(row_layout)
        
        # Store reference
        self.input_rows.append({
            'number_label': number_label,
            'input_field': input_field,
            'row_layout': row_layout
        })

    def get_list_data(self):
        """Returns data in format [(1, "text"), (2, "text"), ...]"""
        return [row['input_field'].text().strip() for row in self.input_rows
                if row['input_field'].text().strip()] 

    def clear_inputs(self):
        """Clears all fields while preserving first row layout"""
        while len(self.input_rows) > 1:
            self._remove_last_row()
        self.input_rows[0]['input_field'].clear()
        self.ensure_single_empty_row()
        self.list_changed.emit()

    def _remove_last_row(self):
        """Removes last row while maintaining layout integrity"""
        if len(self.input_rows) <= 1:
            return
        last_row = self.input_rows.pop()
        self._cleanup_layout(last_row['row_layout'])

    def _cleanup_layout(self, layout):
        """Safely removes layout while preserving your structure"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        layout.deleteLater()
class CreateNewProjectPage(QWidget, Ui_CreateNewProject):
    # Signals
    CreateProjectClicked = Signal()
    project = AnimalPoseTrackerProject()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        self.initialize_controls()

    
    def initialize_controls(self):
        """Initialize all UI controls based on project configuration"""

        self.LocationPathDisplay.setReadOnly(True)

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
        start_dir = str(Path.home())
        selected_dir = QFileDialog.getExistingDirectory(self, 
                                                        "Select Directory", 
                                                        start_dir,
                                                         QFileDialog.ShowDirsOnly)
        if selected_dir:
            self.project.local_path = Path(selected_dir)
            self.LocationPathDisplay.setText(str(self.project.local_path))
        
    def onBrowseSourceDataClicked(self):
        """
        Handles media source selection and updates project config directly.
        Uses local variable 'sources' to temporarily store the selection.
        """
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Configure file dialog
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Media Sources")
        dialog.setDirectory(str(Path.home()))
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png)")

        # Initialize local variable
        sources = None
        
        if dialog.exec():
            selected = dialog.selectedFiles()
            
            # Case 1: Directory selected
            if len(selected) == 1 and Path(selected[0]).is_dir():
                dir_path = Path(selected[0])
                all_files = list(dir_path.glob('*'))
                
                videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
                images = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]
                others = [f for f in all_files if f.suffix.lower() not in VIDEO_EXTS | IMAGE_EXTS]
                
                if videos and not images and not others:
                    sources = sorted(str(f) for f in videos)
                elif images and not videos and not others:
                    sources = str(dir_path)
                else:
                    self.showErrorMessage(
                        "Invalid Directory",
                        f"Contains: {len(videos)} videos, {len(images)} images, {len(others)} other files"
                    )
            
            # Case 2: Files selected
            elif selected:
                invalid_files = [f for f in selected if Path(f).suffix.lower() not in VIDEO_EXTS]
                
                if invalid_files:
                    self.showErrorMessage(
                        "Invalid Files",
                        f"First 3 invalid: {', '.join(invalid_files[:3])}" + 
                        ("..." if len(invalid_files) > 3 else "")
                    )
                else:
                    sources = sorted(selected)
            print(sources)
        
        
    def onClearSourceDataClicked(self):
        """Handler for clearing source data list"""
        # TODO: Clear the source data list widget

    def getAllProjectConfig(self):
        project_config = {
            "project_path": self.project.local_path,
            "project_name": self.ProjectConfig.text(),
            "worker": self.WorkerConfig.text(),
            "model_type": self.ModelTypeSelection.currentText(),
            "model_scale": self.ModelScaleSelection.currentText(),
            "keypoints": self.KeypointConfig.value(),
            "visible": self.VisibleSelection.isChecked(),
            "keypoints_name": self.keypoint_manager.get_list_data(),
            "classes": self.ClassConfig.value(),
            "classes_name": self.class_manager.get_list_data(),
            "source": self.SourceList.currentText()
        }
        if project_config["keypoints"] != len(project_config["keypoints_name"]):
            print("Error: Number of keypoints does not match number of keypoint names")
            return None
        if project_config["classes"] != len(project_config["classes_name"]):
            print("Error: Number of classes does not match number of class names")
            return None
        return project_config

    def onCreateProjectClicked(self):
        self.project.local_path = Path(self.LocationPathDisplay.text())
        project_config = self.getAllProjectConfig() 
        if project_config is None:
            return
        self.project.update_config("project", project_config)
        self.project.create_project()
        self.project.print_project_info()

        if self.CopySourceData.isChecked():
            # TODO: Copy source data to project directory
            pass

        self.CreateProjectClicked.emit()
        self.deleteLater()
        
    def onSourceDataSelectionChanged(self, state):
        """Handler for source data selection state change"""
        # TODO: Update UI based on selection state
    
        
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
    

        
        