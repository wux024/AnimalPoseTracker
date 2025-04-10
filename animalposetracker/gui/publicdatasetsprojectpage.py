from PySide6.QtCore import   QMetaObject, Qt, Slot, Q_ARG, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget, QMainWindow
import os
import yaml
import cv2
import numpy as np
import sys

from .ui_publicdatasetsproject import Ui_PublicDatasetProject


class PublicDatasetProjectPage(QWidget, Ui_PublicDatasetProject):
    CreateProjectClicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        
    def setupConnections(self):
        """Initialize all signal-slot connections for public dataset UI"""
        # Path selection
        self.LocationPathSelection.clicked.connect(self.onLocationPathClicked)
        
        # Project creation
        self.CreateProjectBase.clicked.connect(self.onCreateProjectClicked)
        self.CreateProjectBase.clicked.connect(self.CreateProjectClicked)
        
        # Model selection
        self.ModelTypeSelection.currentIndexChanged.connect(self.onModelTypeChanged)
        self.ModelScaleSelection.currentIndexChanged.connect(self.onModelScaleChanged)
        
        # Dataset selection
        self.DatasetSelection.currentIndexChanged.connect(self.onDatasetChanged)

    # --------------------------------------------------
    # Signal Handlers
    # --------------------------------------------------
    
    def onLocationPathClicked(self):
        """Handler for project location path selection"""
        # TODO: Implement directory selection dialog
        # Should update LocationPathDisplay with selected path
        
    def onCreateProjectClicked(self):
        """Handler for public dataset project creation"""
        # TODO: Validate inputs and create project structure
        # Should use selected dataset and model configurations
        self.close()
        self.deleteLater()
        
    def onModelTypeChanged(self, index):
        """Handler for model type selection change"""
        # TODO: Update available model scales based on selected type
        # May need to enable/disable certain scale options
        
    def onModelScaleChanged(self, index):
        """Handler for model scale selection change"""
        # TODO: Update model configuration based on selected scale
        # May affect performance/accuracy tradeoffs
        
    def onDatasetChanged(self, index):
        """Handler for public dataset selection change"""
        # TODO: Update UI based on selected dataset
        # May need to load dataset-specific configurations
        