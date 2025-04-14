from PySide6.QtCore import  Qt, QSettings, QProcess
from PySide6.QtWidgets import  (QMenu, QApplication, QFileDialog, QMainWindow,
                                 QMessageBox, QTreeWidget, QTreeWidgetItem, 
                                 QTreeWidgetItemIterator)
from PySide6.QtGui import QCursor
import os
import sys
from pathlib import Path
from collections import deque

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
        # set work directory
        os.chdir(Path.cwd())
        self.source_type = 'image' 

    def initialize_controls(self):
        """Initialize the controls of the main page"""
        self.project = AnimalPoseTrackerProject()
        self.config_data = {}
        self.config_type = "project"
        self.process = QProcess()

        # set recent projects
        self.maxRecentProjects = 5
        self.recent_projects = deque(maxlen=self.maxRecentProjects)
        self.recentProjectsMenu = QMenu(self)
        self.actionOpenRecent.setMenu(self.recentProjectsMenu)
        self.loadRecentProjects()
        self.updateRecentProjectsMenu()

        self.AnimalPoseTrackerPage.setCurrentIndex(0)
        self.ConfigureTabPage.setCurrentIndex(0)
        self.ConfigureFile.clear()
        
        self.ConfigureFilePathDisplay.setReadOnly(True)

        self.ConfigureFile.setEditTriggers(QTreeWidget.DoubleClicked | QTreeWidget.EditKeyPressed)
        self.TrainingConfigureEdit.setEditTriggers(QTreeWidget.DoubleClicked | QTreeWidget.EditKeyPressed)
        

    def setupConnections(self):
        # File menu actions
        self.actionCreateNewProject.triggered.connect(self.onCreateNewProject)
        self.actionLoadProject.triggered.connect(self.onLoadProject)
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
        self.ExtractionMethodSelection.currentTextChanged.connect(self.onExtractionMethodChanged)
        self.ClusterStepSetup.valueChanged.connect(self.onClusterStepChanged)
        self.ExtractionAlgorithmSelection.currentTextChanged.connect(self.onExtractionAlgorithmChanged)
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
        self.EditTrainingParameters.clicked.connect(self.onEditOtherParameters)
        self.StartTrain.clicked.connect(self.onStartTrain)
        self.EndTrain.clicked.connect(self.onEndTrain)
        self.TrainingConfigureEdit.itemChanged.connect(self.onOtherConfigureEdited)
        
        # Evaluation tab
        self.EditEvaluationParameters.clicked.connect(self.onEditOtherParameters)
        self.EvaluateConfigure.itemChanged.connect(self.onOtherConfigureEdited)
        self.StartEvaluate.clicked.connect(self.onStartEvaluate)
        self.EndEvaluate.clicked.connect(self.onEndEvaluate)
        
        # Inference tab
        self.EditInferenceParameters.clicked.connect(self.onEditOtherParameters)
        self.InferenceConfigure.itemChanged.connect(self.onOtherConfigureEdited)
        self.SelectionSource.clicked.connect(self.onSelectSource)
        self.StartInference.clicked.connect(self.onStartInference)
        self.EndInference.clicked.connect(self.onEndInference)
        
        # Export tab
        self.EditExportParameters.clicked.connect(self.onEditOtherParameters)
        self.ExportConfigure.itemChanged.connect(self.onOtherConfigureEdited)
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
        self.project.update_config("project", self.sub_page.project.project_config)
        self.project.local_path = self.sub_page.project.local_path
        self.project.load_config_file()
        # set work directory
        os.chdir(self.project.project_path)
        project_config_path = str(self.project.project_path / "project.yaml")
        self.ConfigureFilePathDisplay.setText(project_config_path)
        self.initialize_other_config()
        self.addRecentProject(project_config_path)
        # close sub page
        self.sub_page.close()
        self.sub_page = None     

    def onLoadProject(self):
        """Slot for loading an existing project"""
        # Add your implementation here
        config_path = QFileDialog.getOpenFileName(self, 
                                                  "Select Configuration File", 
                                                  "", "YAML Files (*.yaml *.yml)")[0]
        if config_path:
            self.project.load_project_config(config_path)
            self.AnimalPoseTrackerPage.setCurrentIndex(1)
            project_config_path = str(self.project.project_path / "project.yaml")
            self.ConfigureFilePathDisplay.setText(project_config_path)
            self.initialize_other_config()
            self.addRecentProject(project_config_path)
            # set work directory
            os.chdir(self.project.project_path)

    def initialize_other_config(self):
        """Initialize the other config"""
        self.TrainingConfigureEdit.blockSignals(True)
        self.EvaluateConfigure.blockSignals(True)
        self.InferenceConfigure.blockSignals(True)
        self.ExportConfigure.blockSignals(True)
        iterators = []
        iterators.append(QTreeWidgetItemIterator(self.TrainingConfigureEdit))
        iterators.append(QTreeWidgetItemIterator(self.EvaluateConfigure))
        iterators.append(QTreeWidgetItemIterator(self.InferenceConfigure))
        iterators.append(QTreeWidgetItemIterator(self.ExportConfigure))
        for iterator in iterators:
            while iterator.value():
                    item = iterator.value()
                    key = item.text(0)  # First column contains parameter names
                    # Only update if the key exists in project config
                    if key in self.project.other_config:
                        current_value = self.project.other_config[key]
                        item.setText(1, str(current_value))
                    iterator += 1 
        self.TrainingConfigureEdit.blockSignals(False)
        self.EvaluateConfigure.blockSignals(False)
        self.InferenceConfigure.blockSignals(False)
        self.ExportConfigure.blockSignals(False)

    def loadRecentProjects(self):
        """Load saved projects from QSettings"""
        settings = QSettings("Northeast Normal University", "AnimalPoseTracker")
        projects = settings.value("recentProjects", [], type=list)
        self.recent_projects.extend(p for p in projects if Path(p).exists())

    def updateRecentProjectsMenu(self):
        """Consolidated menu updater"""
        self.recentProjectsMenu.clear()
        
        if not self.recent_projects:
            self.recentProjectsMenu.addAction("No recent projects").setEnabled(False)
        else:
            for i, path in enumerate(self.recent_projects):
                action = self.recentProjectsMenu.addAction(f"&{i+1} {path}")
                action.triggered.connect(lambda _, p=path: self.openRecentProject(p))
        
        self.recentProjectsMenu.addSeparator()
        self.recentProjectsMenu.addAction("Clear All").triggered.connect(self.clearRecentProjects)

    def openRecentProject(self, path):
        """Actually open a project file"""
        try:
            if Path(path).exists():
                # Add your project loading logic here
                self.project.load_project_config(path)
                self.AnimalPoseTrackerPage.setCurrentIndex(1)
                project_config_path = str(self.project.project_path / "project.yaml")
                self.ConfigureFilePathDisplay.setText(project_config_path)
                self.initialize_other_config()
                self.addRecentProject(path)
                return True
            return False
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open project:\n{str(e)}")
            return False

    def addRecentProject(self, project_path):
        """Add with validation"""
        path = str(Path(project_path).absolute())
        if not Path(path).exists():
            return
        
        if path in self.recent_projects:
            self.recent_projects.remove(path)
        
        self.recent_projects.appendleft(path)
        self.saveRecentProjects()
        self.updateRecentProjectsMenu()

    def clearRecentProjects(self):
        self.recent_projects.clear()
        self.saveRecentProjects()
        self.updateRecentProjectsMenu()  # Update UI

    def saveRecentProjects(self):
        settings = QSettings("Northeast Normal University", "AnimalPoseTracker")
        settings.setValue("recentProjects", list(self.recent_projects))

    def onSave(self):
        """Slot for saving the current project"""
        self.project.save_configs()

    def onExit(self):
        """Slot for exiting the application"""
        if self.sub_page is not None:
            self.sub_page.close()
        self.close()

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
        """Change application theme/stylesheets"""
        try:
            if theme == "dark":
                self._applyDarkTheme()
            elif theme == "light": 
                self._applyLightTheme()
            else:
                raise ValueError(f"Unknown theme: {theme}")
            
        except Exception as e:
            QMessageBox.warning(self, "Theme Error", str(e))
    
    def _applyDarkTheme(self):
        """Apply dark mode stylesheet"""
        print("Dark theme applied")

    def _applyLightTheme(self):
        """Reset to default light theme"""
        print("Light theme applied")

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
        elif self.config_type == "other":
            self.config_data = self.project.other_config
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
        self.project._save_config(self.config_type)

    def onCancelConfigureFile(self):
        """Slot for canceling configure file changes"""
        self.ConfigureFile.clear()

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
        """Slot for selecting videos/images
        Handles media source selection with context menu for type selection.
        Maintains single button interaction with intelligent type detection.
        """
        # Create context menu
        menu = QMenu(self)
        video_action = menu.addAction("Select added Video(s)")
        image_action = menu.addAction("Select added Image Directory")
        
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
                self.project.add_source_to_project(valid)
            else:
                self.showErrorMessage("Invalid Selection", "No valid video files selected")
                

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
                self.project.add_source_to_project(dir_path)
        
    def onClearVideosImages(self):
        """Slot for clearing videos/images selection"""
        self.SelectionVideosImagesLabel.setText("0 video or image selected")
        self.project.update_config("project", {"sources", []})

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
    
    def onEditOtherParameters(self):
        """Slot for editing other parameters"""
        button = self.sender()
        BUTTON_TO_WIDGET_MAP = {
        self.EditTrainingParameters: self.TrainingConfigureEdit,
        self.EditEvaluationParameters: self.EvaluateConfigure,
        self.EditInferenceParameters: self.InferenceConfigure,
        self.EditExportParameters: self.ExportConfigure
        }
        tree_widget = BUTTON_TO_WIDGET_MAP[button]
        if tree_widget is None:
            return
        # Determine current mode from button text
        current_text = button.text()
        is_edit_mode = current_text.startswith("Edit")
        parameter_type = current_text.split()[1] 
        iterator = QTreeWidgetItemIterator(tree_widget)
        while iterator.value():
            item = iterator.value()
            if is_edit_mode:
                # EDIT MODE: Enable editing and load current values
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                # SAVE MODE: Disable editing and save changes
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            iterator += 1  # Move to next item

        # Toggle button text for next action
        new_button_text = f"Save {parameter_type} Parameters" if is_edit_mode else f"Edit {parameter_type} Parameters"
        button.setText(new_button_text)

    def onOtherConfigureEdited(self, item, column):
        """Slot function for handling other parameters editing"""
        if column == 1:
            new_value = item.text(1)
            key = item.text(0)
            if key in self.project.other_config:
                self.project.other_config[key] = new_value
            else:
                raise ValueError(f"Invalid key {key} in other config")


    def onStartTrain(self):
        """Slot for starting training"""
        cmd = [
            "yolo",
            "pose",
            "train",
            f"cfg=configs/other.yaml"
        ]
        self.process.setProcessChannelMode(QProcess.ForwardedChannels)
        self.process.start(cmd[0], cmd[1:])
        self.StartTrain.setEnabled(False)
        self.EndTrain.setEnabled(True)

    def onEndTrain(self):
        """Slot for ending training"""
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        self.StartTrain.setEnabled(True)
        self.EndTrain.setEnabled(False)

    def onStartEvaluate(self):
        """Slot for starting evaluation"""
        self.project.update_config("other", {"model": "runs/train/best.pt"})
        self.project.update_config("other", {"name": "val"})
        self.project.save_configs("other")
        cmd = [
            "yolo",
            "pose",
            "val",
            f"cfg=configs/other.yaml"
        ]
        self.process.start(cmd[0], cmd[1:])
        self.StartEvaluate.setEnabled(False)
        self.EndEvaluate.setEnabled(True)


    def onEndEvaluate(self):
        """Slot for ending evaluation"""
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        self.StartEvaluate.setEnabled(True)
        self.EndEvaluate.setEnabled(False)


    def onStartInference(self):
        """Slot for starting inference"""
        self.project.update_config("other", {"model": "runs/train/best.pt"})
        self.project.update_config("other", {"name": "predict"})
        self.project.save_configs("other")
        if self.inference_source is None:
            self.inference_source = Path(self.project.project_config["path"]) / self.project.dataset_config["test"]
            self.inference_source = str(self.inference_source)
        cmd = [
            "yolo",
            "pose",
            "predict",
            f"cfg=configs/other.yaml",
            f"source={self.inference_source}"
        ]
        self.process.start(cmd[0], cmd[1:])
        self.StartInference.setEnabled(False)
        self.EndInference.setEnabled(True)


    def onEndInference(self):
        """Slot for ending inference"""
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        self.StartInference.setEnabled(True)
        self.EndInference.setEnabled(False)
        
    def onSelectSource(self):
        # Create context menu
        menu = QMenu(self)
        file_action = menu.addAction("Select a video/image file")
        path_action = menu.addAction("Select an video/image directory")
        
        # Show menu at cursor position
        action = menu.exec_(QCursor.pos())
        
        if action == file_action:
            self._handleFileSelection()
        elif action == path_action:
            self._handlePathDirSelection()
        
    def _handleFileSelection(self):
        """Show file dialog and return selected file path"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Media File",
            filter="All Supported Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;"
                "Video Files (*.mp4 *.avi *.mov *.mkv);;"
                "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.inference_source = file_path
    
    def _handlePathDirSelection(self):
        """Show directory dialog and return selected directory path"""
        dir_path = QFileDialog.getExistingDirectory(
            parent=self,
            caption="Select Directory"
        )
        if dir_path:
            self.inference_source = dir_path
        
    def onSelectModelWeights(self):
        """Show file dialog and return selected file path"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Media File",
            filter="All Supported Files (*.pt *.onnx);;"
        )
        if file_path:
            self.project.update_config("other", {"model": file_path})

    def onStartExport(self):
        """Slot for starting export"""
        self.project.update_config("other", {"model": "runs/train/best.pt"})
        self.project.update_config("other", {"name": "export"})
        self.project.save_configs("other")
        cmd = [
            "yolo",
            "pose",
            "export",
            f"cfg=configs/other.yaml"
        ]
        self.process.start(cmd[0], cmd[1:])
   

def main():
    app = QApplication(sys.argv)
    main_window = AnimalPoseTrackerPage()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


        