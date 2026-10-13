from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import  QFileDialog, QWidget
from pathlib import Path

from .ui_publicdatasetsproject import Ui_PublicDatasetProject
from .utils import ProjectFormBase
from animalposetracker.gui import LOGO_SMALL_PATH
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS


class PublicDatasetProjectPage(ProjectFormBase, QWidget, Ui_PublicDatasetProject):
    CreateProjectClicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        self.initialize_controls()
    
    def initialize_controls(self):

        # Set here rather than in the Designer file: the icon is a file on disk, and a path baked
        # into `ui_*.py` breaks as soon as the sources are regenerated.
        self.setWindowIcon(QIcon(LOGO_SMALL_PATH))
        self.project = AnimalPoseTrackerProject()
        # The first entry in the list, not a hard-coded name: the page used to open on whatever
        # `DATA_YAML_PATHS` happened to list first, which is also what the combo shows.
        self.dataset = next(iter(DATA_YAML_PATHS))
        self.project.local_path = Path.cwd()
        self.LocationPathDisplay.setText(str(self.project.local_path))

        # Debounce timers for the name fields; see ProjectFormBase.
        self._init_project_form_timers()

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
        
    def onDatasetChanged(self, dataset):
        """Handler for public dataset selection change"""
        if dataset in DATA_YAML_PATHS:
            self.dataset = dataset