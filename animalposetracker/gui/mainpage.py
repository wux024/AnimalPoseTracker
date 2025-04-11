from PySide6.QtCore import   QMetaObject, Qt, Slot, Q_ARG
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import  (QApplication, QFileDialog, QWidget, QMainWindow,
                                 QMessageBox, QTreeWidget, QTreeWidgetItem)
import os
import yaml
import cv2
import numpy as np
import sys
from pathlib import Path


from .ui_animalposetracker import Ui_AnimalPoseTracker
from .inferencepage import AnimalPoseInferencePage
from .createnewprojectpage import CreateNewProjectPage
from .publicdatasetsprojectpage import PublicDatasetProjectPage
from animalposetracker.gui import WindowFactory
from animalposetracker.projector import AnimalPoseTrackerProject

class AnimalPoseTrackerPage(QMainWindow, Ui_AnimalPoseTracker):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)
        self.setupConnections()
        self.setupConstants()
        self.initialize_controls()

    
    def setupConstants(self):
        self.sub_page = None

    def initialize_controls(self):
        """Initialize the controls of the main page"""
        self.project = AnimalPoseTrackerProject()
        self.config_data = {}
        self.config_type = "project"
        self.AnimalPoseTrackerPage.setCurrentIndex(0)
        self.ConfigureTabPage.setCurrentIndex(0)
        self.ConfigureFile.clear()
        
        self.ConfigureFilePathDisplay.setReadOnly(True)

        self.ConfigureFile.setEditTriggers(QTreeWidget.DoubleClicked | QTreeWidget.EditKeyPressed)
        

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
        self.ConfigureFile.itemChanged.connect(self.OnConfigureFileEdited)
        
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
        self.sub_page.CreateProjectClicked.connect(self._OpenConfigureTabPage)
    
    def onCreatePublicDatasetsProject(self):
        """Slot for opening public datasets"""
        if hasattr(self, 'sub_page') and self.sub_page is not None:
            self.sub_page.close()
            self.sub_page = WindowFactory.run(PublicDatasetProjectPage)
            self.sub_page.deleteLater()        
        self.sub_page = WindowFactory.run(PublicDatasetProjectPage)
        self.sub_page.CreateProjectClicked.connect(self._OpenConfigureTabPage)
    
    def _OpenConfigureTabPage(self):
        """Slot for handling the creation of a new project"""
        self.AnimalPoseTrackerPage.setCurrentIndex(1)
        self.project.local_path = self.sub_page.project.local_path
        self.project.update_config("project", self.sub_page.project.project_config)
        self.project.update_config("dataset", self.sub_page.project.dataset_config)
        self.project.update_config("model", self.sub_page.project.model_config)
        self.sub_page.close()
        self.sub_page = None
        project_config_path = str(self.project.project_path / "configs" / "project.yaml")
        self.ConfigureFilePathDisplay.setText(project_config_path)

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

    def onBrowseConfigureFile(self):
        """Slot for browsing and selecting YAML configuration files.
        
        Opens a file dialog to select .yaml or .yml files, validates the selection,
        displays the path in UI, and optionally loads the file content.
        """

        # Initialize file dialog with appropriate settings
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Configuration File")
        file_dialog.setNameFilter("YAML Files (*.yaml *.yml)")
        file_dialog.setFileMode(QFileDialog.ExistingFile) 
        
        # Execute dialog and process selection
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:  # Check if user selected a file
                file_path = selected_files[0]
                
                # Validate file extension (recommended safety check)
                if not file_path.lower().endswith(('.yaml', '.yml')):
                    QMessageBox.warning(
                        self, 
                        "Invalid File", 
                        "Please select a valid YAML file (.yaml or .yml extension)"
                    )
                    return
                    
                # Update UI with selected path
                self.ConfigureFilePathDisplay.setText(file_path)
                
                # Optional: Load and parse the YAML file
                # try:
                #     with open(file_path, 'r') as f:
                #         config = yaml.safe_load(f)  # Safe loading prevents code execution
                #         print("Successfully loaded config:", config)
                        
                #         # Here you can add additional config processing logic
                #         # For example: self.process_config(config)
                        
                # except yaml.YAMLError as e:
                #     QMessageBox.critical(
                #         self, 
                #         "YAML Error", 
                #         f"Invalid YAML syntax:\n{str(e)}"
                #     )
                # except IOError as e:
                #     QMessageBox.critical(
                #         self, 
                #         "File Error", 
                #         f"Cannot read file:\n{str(e)}"
                #     )
                # except Exception as e:
                #     QMessageBox.critical(
                #         self, 
                #         "Unexpected Error", 
                #         f"Failed to process file:\n{str(e)}"
                #     )

    def onEditConfigureFile(self):
        """Slot for file"""
        yaml_file = self.ConfigureFilePathDisplay.text()
        self.config_type = Path(yaml_file).stem.lower()
        if self.config_type not in self.project.CONFIG_TYPES:
            QMessageBox.warning(
                self, 
                "Invalid Configuration Type", 
                "Please select a valid configuration type (project, dataset, or model)"
            )
        if self.config_type == "project":
            self.config_data = self.project.project_config
        elif self.config_type == "dataset":
            self.config_data = self.project.dataset_config
        elif self.config_type == "model":
            self.config_data = self.project.model_config
        self.DisplayConfigureFile()

    def DisplayConfigureFile(self):
        self.ConfigureFile.clear()
        self.AddItemsToTree(self.config_data)
    
    def AddItemsToTree(self, data, parent=None, path=None):
        if parent is None:
            parent = self.ConfigureFile
            path = []
        if isinstance(data, dict):
            for key, value in data.items():
                item = QTreeWidgetItem(parent, [str(key), ""])
                self.AddItemsToTree(value, item, path + [("dict", key)]) 
        elif isinstance(data, list):
            for index, value in enumerate(data):
                item = QTreeWidgetItem(parent, [f"[{index}]", ""])
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.AddItemsToTree(value, item, path + [("list", index)]) 
        else:
            if parent.columnCount() < 2:
                parent.setText(1, str(data))
                parent.setFlags(parent.flags() | Qt.ItemIsEditable)
                parent.setData(1, Qt.UserRole, path)
            else:
                parent.setText(1, str(data))
                parent.setFlags(parent.flags() | Qt.ItemIsEditable)
                parent.setData(1, Qt.UserRole, path)
    
    def OnConfigureFileEdited(self, item, column):
        if column == 1:
            new_value = item.text(1)
            path = item.data(1, Qt.UserRole)
            if path:            
                self.update_config_data(self.config_data, path, new_value)

    def update_config_data(self, data, path, new_value):
        current = data
        for step in path[:-1]:
            if step[0] == "dict":
                current = current[step[1]]
            elif step[0] == "list":
                current = current[step[1]]
        
        last_step = path[-1]
        if last_step[0] == "dict":
            current[last_step[1]] = self.parse_value(new_value)
        elif last_step[0] == "list":
            current[last_step[1]] = self.parse_value(new_value)

    def parse_value(self, value_str):
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                if value_str.lower() == "true":
                    return True
                elif value_str.lower() == "false":
                    return False
                else:
                    return value_str 

    def onSaveConfigureFile(self):
        """Slot for saving configure file"""
        print(f"{self.config_type} Save Configure File clicked")
        self.project._save_config(self.config_type)

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


        