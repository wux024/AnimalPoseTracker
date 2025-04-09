from PySide6.QtCore import   QMetaObject, Qt, Slot, Q_ARG
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  QApplication, QFileDialog, QWidget, QMainWindow
import os
import yaml
import cv2
import numpy as np
import sys


from .ui_animalposetracker import Ui_AnimalPoseTracker
from .inferencepage import AnimalPoseInferencePage
from .createnewprojectpage import CreateNewProjectPage
from .publicdatasetsprojectpage import PublicDatasetProjectPage
from animalposetracker.gui import WindowFactory

class AnimalPoseTrackerPage(QMainWindow, Ui_AnimalPoseTracker):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)
        self.setupConnections()
        self.setupConstants()
        self.AnimalPoseTrackerPage.setCurrentIndex(0)

    
    def setupConstants(self):
        self.sub_page = None

    def setupConnections(self):
        # File menu actions
        self.actionCreateNewProject.triggered.connect(self.onCreateNewProject)
        self.actionLoadProject.triggered.connect(self.onLoadProject)
        self.actionOpenRecent.triggered.connect(self.onOpenRecent)
        self.actionSave.triggered.connect(self.onSave)
        self.actionExit.triggered.connect(self.onExit)
        
        # Tools menu actions
        self.actionAnnotator.triggered.connect(self.onOpenAnnotator)
        self.actionInferencer.triggered.connect(self.onOpenInferencer)
        self.actionTracker.triggered.connect(self.onOpenTracker)
        
        # Help menu actions
        self.actionDoc.triggered.connect(self.onOpenDocumentation)
        self.actionCheckUpdated.triggered.connect(self.onCheckUpdates)
        self.actionHelp.triggered.connect(self.onHelp)
        
        # View menu actions
        self.actionDark.triggered.connect(lambda: self.onChangeTheme('dark'))
        self.actionLight.triggered.connect(lambda: self.onChangeTheme('light'))
        
        # Public datasets
        self.actionPublicDatasetsProject.triggered.connect(self.onCreatePublicDatasetsProject)
        
        # Main page buttons
        self.CreateNewProjectButton.clicked.connect(self.onCreateNewProject)
        self.LoadProjectButton.clicked.connect(self.onLoadProject)
        self.PublicDatasetsProjectButton.clicked.connect(self.onCreatePublicDatasetsProject)
        
        # Configure file operations
        self.ConfigureFilePathBrowser.clicked.connect(self.onBrowseConfigureFile)
        self.ConfigureFileEdit.clicked.connect(self.onEditConfigureFile)
        self.SaveConfigureFile.clicked.connect(self.onSaveConfigureFile)
        self.CannelConfigureFile.clicked.connect(self.onCancelConfigureFile)
        
        # Extract and Label Frames tab
        self.ExtractionMethodSelection.currentIndexChanged.connect(self.onExtractionMethodChanged)
        self.ClusterStepSetup.valueChanged.connect(self.onClusterStepChanged)
        self.ExtractionAlgorithmSelection.currentIndexChanged.connect(self.onExtractionAlgorithmChanged)
        self.SelectionVideosImages.clicked.connect(self.onSelectVideosImages)
        self.ClearVideosImages.clicked.connect(self.onClearVideosImages)
        self.ExtracFrames.clicked.connect(self.onExtractFrames)
        
        # frames section
        self.SaveYOLO.stateChanged.connect(self.onYOLOFormatChanged)
        self.SaveCOCO.stateChanged.connect(self.onCOCOFormatChanged)
        self.StartLabelFrames.clicked.connect(self.onStartLabelFrames)
        self.CheckLabelledFrames.clicked.connect(self.onCheckLabelledFrames)
        self.BuildSkeleton.clicked.connect(self.onBuildSkeleton)
        
        # Training tab
        self.ResumeTrain.clicked.connect(self.onResumeTrain)
        self.EditTrainingParameters.clicked.connect(self.onEditTrainingParameters)
        self.StartTrain.clicked.connect(self.onStartTrain)
        self.EndTrain.clicked.connect(self.onEndTrain)
        
        # Evaluation tab
        self.EditEvaluationParameters.clicked.connect(self.onEditEvaluationParameters)
        self.StartEvaluate.clicked.connect(self.onStartEvaluate)
        self.EndEvaluate.clicked.connect(self.onEndEvaluate)
        
        # Inference tab
        self.EditInferenceParameters.clicked.connect(self.onEditInferenceParameters)
        self.SelectionSource.clicked.connect(self.onSelectSource)
        self.StartInference.clicked.connect(self.onStartInference)
        self.EndInference.clicked.connect(self.onEndInference)
        
        # Export tab
        self.EditExportParameters.clicked.connect(self.onEditExportParameters)
        self.StartModelWeights.clicked.connect(self.onSelectModelWeights)
        self.StartExport.clicked.connect(self.onStartExport)

    def onCreateNewProject(self):
        """Slot for creating a new project"""
        if hasattr(self, 'sub_page') and self.sub_page is not None:
            self.sub_page.close()
            self.sub_page = WindowFactory.run(CreateNewProjectPage)
            self.sub_page.deleteLater()
        
        self.sub_page = WindowFactory.run(CreateNewProjectPage)
        self.sub_page.CreateProjectClicked.connect(self.handleCreateNewProject)
    
    def handleCreateNewProject(self):
        """Slot for handling the creation of a new project"""
        self.AnimalPoseTrackerPage.setCurrentIndex(1)

    def onLoadProject(self):
        """Slot for loading an existing project"""
        print("Load Project clicked")
        # Add your implementation here

    def onOpenRecent(self):
        """Slot for opening a recent project"""
        print("Open Recent clicked")
        # Add your implementation here

    def onSave(self):
        """Slot for saving the current project"""
        print("Save clicked")
        # Add your implementation here

    def onExit(self):
        """Slot for exiting the application"""
        print("Exit clicked")
        # Add your implementation here

    def onOpenAnnotator(self):
        """Slot for opening the annotator tool"""
        print("Annotator clicked")
        # Add your implementation here

    def onOpenInferencer(self):
        """Slot for opening the inferencer tool"""
        print("Inferencer clicked")
        # Add your implementation here
        self.sub_page = AnimalPoseInferencePage()

    def onOpenTracker(self):
        """Slot for opening the tracker tool"""
        print("Tracker clicked")
        # Add your implementation here

    def onOpenDocumentation(self):
        """Slot for opening documentation"""
        print("Documentation clicked")
        # Add your implementation here

    def onCheckUpdates(self):
        """Slot for checking for updates"""
        print("Check Updates clicked")
        # Add your implementation here

    def onHelp(self):
        """Slot for opening help"""
        print("Help clicked")
        # Add your implementation here

    def onChangeTheme(self, theme):
        """Slot for changing the application theme"""
        print(f"Change theme to {theme}")
        # Add your implementation here

    def onCreatePublicDatasetsProject(self):
        """Slot for opening public datasets"""
        print("Create Public Datasets Project clicked")
        self.sub_page = WindowFactory.run(PublicDatasetProjectPage)

    def onBrowseConfigureFile(self):
        """Slot for browsing configure file"""
        print("Browse Configure File clicked")
        # Add your implementation here

    def onEditConfigureFile(self):
        """Slot for file"""
        print("Edit Configure File clicked")
        # Add your implementation here

    def onSaveConfigureFile(self):
        """Slot for saving configure file"""
        print("Save Configure File clicked")
        # Add your implementation here

    def onCancelConfigureFile(self):
        """Slot for canceling configure file changes"""
        print("Cancel Configure File clicked")
        # Add your implementation here

    def onExtractionMethodChanged(self, index):
        """Slot for extraction method selection changed"""
        print(f"Extraction method changed to index {index}")
        # Add your implementation here

    def onClusterStepChanged(self, value):
        """Slot for cluster step value changed"""
        print(f"Cluster step changed to {value}")
        # implementation here

    def onExtractionAlgorithmChanged(self, index):
        """Slot for extraction algorithm selection changed"""
        print(f"Extraction algorithm changed to index {index}")
        # Add your implementation here

    def onSelectVideosImages(self):
        """Slot for selecting videos/images"""
        print("Select Videos/Images clicked")
        # Add your implementation here

    def onClearVideosImages(self):
        """Slot for clearing videos/images selection"""
        print("Clear Videos/Images clicked")
        # Add your implementation here

    def onExtractFrames(self):
        """Slot for extracting frames"""
        print("Extract Frames clicked")
        # Add your implementation here

    def onYOLOFormatChanged(self, state):
        """Slot for YOLO format checkbox state changed"""
        print(f"YOLO format state changed to {state}")
        # Add your implementation here

    def onCOCOFormatChanged(self, state):
        """Slot for COCO format checkbox state changed"""
        print(f"COCO format state changed to {state}")
        # Add your implementation here

    def onStartLabelFrames(self):
        """Slot for starting frame labeling"""
        print("Start Label Frames clicked")
        # Add your implementation here

    def onCheckLabelledFrames(self):
        """Slot for checking labelled frames"""
        print("Check Labelled Frames clicked")
        # Add your implementation here

    def onBuildSkeleton(self):
        """Slot for building skeleton"""
        print("Build Skeleton clicked")
        # Add your implementation here

    def onResumeTrain(self):
        """Slot for resuming training"""
        print("Resume Train clicked")
        # Add your implementation here

    def onEditTrainingParameters(self):
        """Slot for editing training parameters"""
        print("Edit Training Parameters clicked")
        # Add your implementation here

    def onStartTrain(self):
        """Slot for starting training"""
        print("Start Train clicked")
        # Add your implementation here

    def onEndTrain(self):
        """Slot for ending training"""
        print("End Train clicked")
        # Add your implementation here

    def onEditEvaluationParameters(self):
        """Slot for editing evaluation parameters"""
        print("Edit Evaluation Parameters clicked")
        # Add your implementation here

    def onStartEvaluate(self):
        """Slot for starting evaluation"""
        print("Start Evaluate clicked")
        # Add your implementation here

    def onEndEvaluate(self):
        """Slot for ending evaluation"""
        print("End Evaluate clicked")
        # Add your implementation here

    def onEditInferenceParameters(self):
        """Slot for editing inference parameters"""
        print("Edit Inference Parameters clicked")
        # Add your implementation here

    def onSelectSource(self):
        """Slot for selecting inference source"""
        print("Select Source clicked")
        # Add your implementation here

    def onStartInference(self):
        """Slot for starting inference"""
        print("Start Inference clicked")
        # Add your implementation here

    def onEndInference(self):
        """Slot for ending inference"""
        print("End Inference clicked")
        # Add your implementation here

    def onEditExportParameters(self):
        """Slot for editing export parameters"""
        print("Edit Export Parameters clicked")
        # Add your implementation here

    def onSelectModelWeights(self):
        """Slot for selecting model weights"""
        print("Select Model Weights clicked")
        # Add your implementation here

    def onStartExport(self):
        """Slot for starting export"""
        print("Start Export clicked")
        # Add your implementation here
   

def main():
    app = QApplication(sys.argv)
    main_window = AnimalPoseTrackerPage()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


        