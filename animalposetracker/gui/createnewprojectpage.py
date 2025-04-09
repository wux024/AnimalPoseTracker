from PySide6.QtCore import   QMetaObject, Qt, Slot, Q_ARG, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget, QMainWindow
import os
import yaml
import cv2
import numpy as np
import sys

from .ui_createnewproject import Ui_CreateNewProject


class CreateNewProjectPage(QWidget, Ui_CreateNewProject):
    CreateProjectClicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()

    def setupConstants(self):
        """Initialize all constants for this UI"""
        # TODO: Implement constants
        self.project = None
        self.worker = None
        self.project_dir = None
        self.model_type = None
        self.model_scale = None
        self.keypoints = 0
        self.classes = 0
        self.keypoints_names = []
        self.classes_names = []

    def setupConnections(self):
        """Initialize all signal-slot connections for this UI"""
        # Path selection
        self.LocationPathSelection.clicked.connect(self.onLocationPathClicked)
        
        # Source data operations
        self.BrowseSourceData.clicked.connect(self.onBrowseSourceDataClicked)
        self.ClearSourceDataList.clicked.connect(self.onClearSourceDataClicked)
        
        # Project creation
        self.CreateProjectBase.clicked.connect(self.CreateProjectClicked)
        self.CreateProjectBase.clicked.connect(self.onCreateProjectClicked)
        
        # Checkbox states
        self.SourceDataSelected.stateChanged.connect(self.onSourceDataSelectionChanged)
        self.CopySourceData.stateChanged.connect(self.onCopySourceDataChanged)
        
        # Configuration changes
        self.KeypointConfig.valueChanged.connect(self.onKeypointConfigChanged)
        self.ClassConfig.valueChanged.connect(self.onClassConfigChanged)
        
        # Model selection
        self.ModelTypeSelection.currentIndexChanged.connect(self.onModelTypeChanged)
        self.ModelScaleSelection.currentIndexChanged.connect(self.onModelScaleChanged)


    
    def onLocationPathClicked(self):
        """Handler for project location path selection"""
        # TODO: Implement directory selection dialog
        
    def onBrowseSourceDataClicked(self):
        """Handler for browsing source data files"""
        # TODO: Implement file dialog for images/videos
        
    def onClearSourceDataClicked(self):
        """Handler for clearing source data list"""
        # TODO: Clear the source data list widget
        
    def onCreateProjectClicked(self):
        """Handler for project creation"""
        # TODO: Validate inputs and create project structure
        self.CreateProjectClicked.emit()
        self.close()
        self.deleteLater()
        
    def onSourceDataSelectionChanged(self, state):
        """Handler for source data selection state change"""
        # TODO: Update UI based on selection state
        
    def onCopySourceDataChanged(self, state):
        """Handler for copy source data checkbox state change"""
        # TODO: Update copy operation preference
        
    def onKeypointConfigChanged(self, value):
        """Handler for keypoint count configuration change"""
        # TODO: Dynamically adjust keypoint name fields
        
    def onClassConfigChanged(self, value):
        """Handler for class count configuration change"""
        # TODO: Dynamically adjust class name fields
        
    def onModelTypeChanged(self, index):
        """Handler for model type selection change"""
        # TODO: Update available model scales based on type
        
    def onModelScaleChanged(self, index):
        """Handler for model scale selection change"""
        # TODO: Update model configuration based on scale
        