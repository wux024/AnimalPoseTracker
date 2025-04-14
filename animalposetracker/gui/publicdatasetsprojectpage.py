from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import  QFileDialog, QWidget
from pathlib import Path

from .ui_publicdatasetsproject import Ui_PublicDatasetProject
from animalposetracker.projector import AnimalPoseTrackerProject
from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS


class PublicDatasetProjectPage(QWidget, Ui_PublicDatasetProject):
    CreateProjectClicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        self.initialize_controls()
    
    def initialize_controls(self):

        self.project = AnimalPoseTrackerProject()
        self.dataset = "AP10K"
        self.project.local_path = Path.cwd()
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

        self.DatasetSelection.clear()
        self.DatasetSelection.addItems(list(DATA_YAML_PATHS.keys()))
        
    def setupConnections(self):
        """Initialize all signal-slot connections for public dataset UI"""

        self.ProjectConfig.textEdited.connect(self.onProjectTextEdited)
        self.WorkerConfig.textEdited.connect(self.onWorkerTextEdited)
        # Path selection
        self.LocationPathSelection.clicked.connect(self.onLocationPathClicked)
        
        # Project creation
        self.CreateProjectBase.clicked.connect(self.onCreateProjectClicked)
        
        # Model selection
        self.ModelTypeSelection.currentTextChanged.connect(self.onModelTypeChanged)
        self.ModelScaleSelection.currentTextChanged.connect(self.onModelScaleChanged)
        
        # Dataset selection
        self.DatasetSelection.currentTextChanged.connect(self.onDatasetChanged)

    
    def onLocationPathClicked(self):
        """Handler for project location path selection"""
        start_dir = str(Path.home())
        selected_dir = QFileDialog.getExistingDirectory(self, 
                                                        "Select Directory", 
                                                        start_dir,
                                                         QFileDialog.ShowDirsOnly)
        if selected_dir:
            self.project.local_path = Path(selected_dir)
            self.LocationPathDisplay.setText(str(self.project.local_path))
    
    def getAllProjectConfig(self):
        project_config = {
            "project_path": str(self.project.project_path),
            "project_name": self.ProjectConfig.text(),
            "worker": self.WorkerConfig.text(),
            "model_type": self.ModelTypeSelection.currentText(),
            "model_scale": self.ModelScaleSelection.currentText(),
        }

        return project_config
        
    def onCreateProjectClicked(self):
        """Handler for public dataset project creation"""
        # Should use selected dataset and model configurations
        project_config = self.getAllProjectConfig()
        if project_config is None:
            return
        self.project.update_config("project", project_config)
        self.project.local_path = Path(self.LocationPathDisplay.text())
        self.project.create_public_dataset_project(self.dataset)
        self.CreateProjectClicked.emit()
        
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
        
    def onDatasetChanged(self, dataset):
        """Handler for public dataset selection change"""
        if dataset in DATA_YAML_PATHS:
            self.dataset = dataset

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