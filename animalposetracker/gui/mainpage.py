from PySide6.QtCore import  Qt, QSettings, QFile, QTextStream, QTimer
from PySide6.QtWidgets import  (QMenu, QApplication, QFileDialog, QMainWindow,
                                 QMessageBox, QTreeWidget, QTreeWidgetItem, QLabel, 
                                 QTreeWidgetItemIterator, QSplashScreen, QProgressBar,
                                 QDialog)
from PySide6.QtGui import QCursor, QPixmap
import os
import sys
from pathlib import Path
from collections import deque

from .ui_animalposetracker import Ui_AnimalPoseTracker
from .inferencepage import AnimalPoseInferencePage
from .createnewprojectpage import CreateNewProjectPage
from .publicdatasetsprojectpage import PublicDatasetProjectPage
from .annotatorpage import AnimalPoseAnnotatorPage
from .utils import (DatasetSplitDialog, split_dataset_sklearn, 
                    create_dataset_structure, copy_split_files,
                    create_coco_annotations)
from animalposetracker.gui import WindowFactory
from animalposetracker.projector import AnimalPoseTrackerProject
from animalposetracker.utils import (convert_labels_to_coco, 
                                     convert_labels_to_yolo,
                                     KeyframeExtractor)
from animalposetracker.gui import (DARK_THEME_PATH, LIGHT_THEME_PATH, 
                                   LOGO_PATH_TRANSPARENT, LOGO_PATH, WELCOME_PATH)

class AnimalPoseTrackerPage(QMainWindow, Ui_AnimalPoseTracker):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.showMaximized()
        self.initialize_controls()
        

    def initialize_controls(self):
        """Initialize the controls of the main page"""
        if sys.platform == "darwin":
            self.setUnifiedTitleAndToolBarOnMac(True)

        self.sub_page = None
        # set work directory
        os.chdir(Path.cwd())
        self.source_type = 'image' 
        self.project = AnimalPoseTrackerProject()
        self.extractor = KeyframeExtractor(Path.home())
        self.config_data = {}
        self.config_type = "project"
        self.current_mode = None
        self.inference_source = None

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

        self.onChangeTheme("light")
        self.SelectionVideosImagesLabel.setText("0 video or image selected")

        self.setupConnections()
        

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
        self.SelectionVideosImages.clicked.connect(self.onSelectVideosImages)
        self.ClearVideosImages.clicked.connect(self.onClearVideosImages)
        self.ExtracFrames.clicked.connect(self.onExtractFrames)
        
        # frames section
        self.SaveYOLO.setChecked(True)
        self.SaveYOLO.setEnabled(False)
        self.StartLabelFrames.clicked.connect(self.onStartLabelFrames)
        self.StartCreateDatasets.clicked.connect(self.onCreateDatasets)
        self.CheckDatasets.clicked.connect(self.onCheckDatasets)
        
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
        if self.project is not None:
            self.project.save_configs()
        self.close()

    def onOpenAnnotator(self):
        """Slot for opening the annotator tool"""
        if hasattr(self, 'sub_page') and self.sub_page is not None:
            self.sub_page.close()
            self.sub_page = WindowFactory.run(AnimalPoseAnnotatorPage)
            self.sub_page.deleteLater()
        self.sub_page = WindowFactory.run(AnimalPoseAnnotatorPage)

    def onOpenInferencer(self):
        """Slot for opening the inferencer tool"""
        # Add your implementation here
        if hasattr(self, 'sub_page') and self.sub_page is not None:
            self.sub_page.close()
            self.sub_page = WindowFactory.run(AnimalPoseInferencePage)
            self.sub_page.deleteLater()
        self.sub_page = WindowFactory.run(AnimalPoseInferencePage)
    
    def onOpenTracker(self):
        """Slot for opening the tracker tool"""
        self.statusBar().showMessage("Tracker clicked")
        # Add your implementation here

    def onOpenDocumentation(self):
        """Slot for opening documentation"""
        self.statusBar().showMessage("Documentation clicked")
        # Add your implementation here

    def onCheckUpdates(self):
        """Slot for checking for updates"""
        self.statusBar().showMessage("Check updates clicked")
        # Add your implementation here

    def onHelp(self):
        """Slot for opening help"""
        self.statusBar().showMessage("Help clicked")
        # Add your implementation here

    def onChangeTheme(self, theme):
        """Change application theme/stylesheets"""
        theme_file = {
            "dark": DARK_THEME_PATH,
            "light": LIGHT_THEME_PATH
            }.get(theme)
        if not theme_file:
            QMessageBox.warning(self, "Error", "Invalid theme name")
            return
        try:
            file = QFile(theme_file)
            if file.open(QFile.ReadOnly | QFile.Text):
                stream = QTextStream(file)
                self.setStyleSheet(stream.readAll())
                file.close()
            if theme == "dark":
                self.Logo.setPixmap(QPixmap(LOGO_PATH))
            elif theme == "light":
                self.Logo.setPixmap(QPixmap(LOGO_PATH_TRANSPARENT))
        except Exception as e:
            QMessageBox.warning(self, "Theme Error", str(e))
        
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
            current[last_step[1]] = self.convert_value(new_value)
        elif last_step[0] == "list":
            current[last_step[1]] = self.convert_value(new_value)

    def onSaveConfigureFile(self):
        """Slot for saving configure file"""
        self.project._save_config(self.config_type)

    def onCancelConfigureFile(self):
        """Slot for canceling configure file changes"""
        self.ConfigureFile.clear()

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
                self.project.add_source_to_project(valid, move_or_copy='copy')
                self.SelectionVideosImagesLabel.setText(f"{len(valid)} video(s) selected")
            else:
                QMessageBox.warning(self, "No Valid Videos", "No valid video files selected")
                

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
                QMessageBox.warning(self, 
                                    "No Valid Images", 
                                    "No valid image files found in directory")
            elif non_images:
                QMessageBox.warning(self, 
                                    "Non-Image Files", 
                                    "Non-image files found in directory")
            else:
                self.project.add_source_to_project(dir_path)
                self.SelectionVideosImagesLabel.setText(f"{len(images)} images selected")
        
    def onClearVideosImages(self):
        """Slot for clearing videos/images selection"""
        self.SelectionVideosImagesLabel.setText("0 video or image selected")
        self.project.update_config("project", {"sources", []})

    def onExtractFrames(self):
        """Slot for extracting frames"""
        self.statusBar().showMessage("Starting frame extraction...")
        self.extractor.output_dir = self.project.project_path / "sources" / "extracted"
        extract_mode = self.ExtractionMethodSelection.currentText()
        extract_algorithm = self.ExtractionAlgorithmSelection.currentText()
        sources = self.project.project_config.get("sources", [])
        n_clusters = self.ClusterStepSetup.value()
        interval = self.SampleIntervalSetup.value()
        if not sources:
            QMessageBox.warning(self, "No Sources", "No sources found in project.yaml")
            return
        self.source_type = "image" if Path(sources[0]).is_dir() else "video"
        if self.source_type == "video":
            self.extractor.extract_keyframes_from_videos(
                video_paths=sources, 
                mode=extract_mode,
                algorithm=extract_algorithm,
                n_clusters=n_clusters,
                interval=interval 
            )
        elif self.source_type == "image":
            self.extractor.extract_keyframes_from_images(
                image_path=sources[0], 
                mode=extract_mode,
                algorithm=extract_algorithm,
                n_clusters=n_clusters,
                interval=interval 
            )
        self.statusBar().showMessage("Frame extraction complete.")

    def onStartLabelFrames(self):
        """Slot for starting frame labeling"""
        if hasattr(self, 'sub_page') and self.sub_page is not None:
            self.sub_page.close()
            self.sub_page = WindowFactory.run(AnimalPoseAnnotatorPage)
            self.sub_page.deleteLater()
        self.sub_page = WindowFactory.run(AnimalPoseAnnotatorPage)

    def onCreateDatasets(self):
        """Slot for starting dataset creation"""
        extracted_path = self.project.project_path / "sources" / "extracted"
        if not extracted_path.exists():
            QMessageBox.warning(self, 
                                "No Extracted Frames", 
                                "No extracted frames found. Please extract frames first.")
            return
        yolo_path = extracted_path / "yolo_format"
        coco_path = extracted_path / "coco_format"
        if not yolo_path.exists() and not coco_path.exists():
            self.statusBar().showMessage("Stating export process...")
            if self.SaveCOCO.isChecked():
                convert_labels_to_coco(extracted_path, self.project.project_config)
                self.statusBar().showMessage("COCO dataset export complete.")
            if self.SaveYOLO.isChecked():
                convert_labels_to_yolo(extracted_path)
                self.statusBar().showMessage("YOLO dataset export complete.")
            self.statusBar().showMessage("Dataset export complete.")
        self._SplitDataset()
        
    def _SplitDataset(self):
        """Main function to handle dataset splitting"""
        # Show ratio dialog
        dialog = DatasetSplitDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        # Proceed with dataset creation
        self._CreateDatasetsWithSplit(dialog.ratios)

    def _CreateDatasetsWithSplit(self, ratios):
        """Main function that uses proper sklearn splitting"""
        # 1. Validate inputs and prepare paths
        extracted_path = self.project.project_path / "sources" / "extracted"
        yolo_path = extracted_path / "yolo_format"
        
        if not yolo_path.exists():
            QMessageBox.warning(self, "Error", "YOLO format labels not found.")
            return
        
        # 2. Get all valid image-label pairs
        image_files = sorted(extracted_path.glob("*.*"))  # Get all files
        image_files = [f for f in image_files if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        label_files = [yolo_path / f"{f.stem}.txt" for f in image_files]
        
        # Filter only pairs where both files exist
        valid_pairs = [(img, lbl) for img, lbl in zip(image_files, label_files) if lbl.exists()]
        if not valid_pairs:
            QMessageBox.warning(self, "Error", "No valid image-label pairs found.")
            return
        
        # 3. Split using sklearn (unpack the pairs)
        image_paths, label_paths = zip(*valid_pairs)
        train, val, test = split_dataset_sklearn(image_paths, label_paths, ratios)

        # 4. Create directory structure
        dataset_path = self.project.project_path / "datasets"
        dirs = create_dataset_structure(dataset_path)
        
        # 5. Copy files to their destinations
        copy_split_files(zip(*train), dirs['images_train'], dirs['labels_train'])
        copy_split_files(zip(*val), dirs['images_val'], dirs['labels_val'])
        copy_split_files(zip(*test), dirs['images_test'], dirs['labels_test'])
        
        # 6. Handle COCO format if needed
        if hasattr(self, 'SaveCOCO') and self.SaveCOCO.isChecked():
            coco_path = extracted_path / "coco_format" / "coco_format.json"
            create_coco_annotations(coco_path, 
                                    dirs['annotations'], 
                                    train, val, test)
        
        QMessageBox.information(self, "Success", "Dataset splitting completed!")

    def onCheckDatasets(self):
        """
        Verify that every image in the dataset has a corresponding YOLO annotation file
        """
        
        # 1. Get paths to dataset directories
        dataset_path = self.project.project_path / "datasets"
        yolo_labels_path = dataset_path / "labels"
        
        # 2. Check all splits (train/val/test)
        splits = ["train", "val", "test"]
        issues_found = 0
        
        for split in splits:
            self.statusBar().showMessage(f"Checking {split} split...")
            # Get all image files in the split
            image_dir = dataset_path / "images" / split
            if not image_dir.exists():
                self.statusBar().showMessage(f"No {split} split found.")
                continue
                
            image_files = list(image_dir.glob("*.*"))
            image_files = [f for f in image_files if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            
            # Get corresponding label files
            label_dir = yolo_labels_path / split
            missing_labels = []
            empty_labels = []
            
            for img_path in image_files:
                label_path = label_dir / f"{img_path.stem}.txt"
                
                # Case 1: Label file missing
                if not label_path.exists():
                    missing_labels.append(img_path.name)
                    issues_found += 1
                    continue
                    
                # Case 2: Label file exists but is empty
                if label_path.stat().st_size == 0:
                    empty_labels.append(img_path.name)
                    issues_found += 1
                    
            # Print results for current split
            if missing_labels:
                print(f"Missing YOLO labels for {len(missing_labels)} images:")
                for name in missing_labels[:5]:  # Show first 5 examples
                    print(f"  - {name}")
                if len(missing_labels) > 5:
                    print(f"  (...and {len(missing_labels)-5} more)")
                    
            if empty_labels:
                print(f"Empty YOLO labels for {len(empty_labels)} images:")
                for name in empty_labels[:5]:
                    print(f"  - {name}")
                if len(empty_labels) > 5:
                    print(f"  (...and {len(empty_labels)-5} more)")
                    
            if not missing_labels and not empty_labels:
                self.statusBar().showMessage(f"All {len(image_files)} images have valid YOLO annotations")
        
        # 3. Show summary
        if issues_found:
            msg = f"Found {issues_found} issues in dataset annotations!"
            print(msg)
            QMessageBox.warning(self, "Dataset Issues", msg)
        else:
            msg = "All dataset splits are valid - every image has a proper YOLO annotation"
            QMessageBox.information(self, "Dataset Check", msg)

    def onResumeTrain(self):
        """Slot for resuming training"""
        file = QFileDialog.getOpenFileName(self, 
                                            "Select Resuming Model", 
                                            "", "PyTorch Model (*.pt)")[0]
        if file:
            resume_path = file
        else:
            resume_path = str(self.project.project_path / "runs/train/weights/last.pt")
        self.ResumeTrain.setEnabled(False)
        self.StartTrain.setEnabled(False)
        self.EndTrain.setEnabled(True)
        self.project.resume(resume_path)

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
        
        if not is_edit_mode:
            # Save changes to config file
            self.project.save_configs("other")

        # Toggle button text for next action
        new_button_text = f"Save {parameter_type} Parameters" if is_edit_mode else f"Edit {parameter_type} Parameters"
        button.setText(new_button_text)

    def convert_value(self, value):
        """
        Converts a string value to appropriate Python type.
        
        Handles basic types: None, bool, int, float, str
        Does NOT handle scientific notation or complex types.
        
        Args:
            value: Input value (str or any type)
            
        Returns:
            Appropriate Python type (None, bool, int, float, or original type)
            
        Examples:
            >>> convert_value("123") → 123
            >>> convert_value("3.14") → 3.14 
            >>> convert_value("true") → True
            >>> convert_value("Null") → None
            >>> convert_value(123) → 123 (non-strings returned as-is)
        """
        # Return non-strings immediately
        if not isinstance(value, str):
            return value
            
        value = value.strip()  # Normalize input
        
        # Handle empty/None cases
        if not value or value in ("None", "Null", "null", "none"):
            return None
            
        # Handle booleans
        if value in ("True", "False", "true", "false"):
            return value == "True" or value == "true"
            
        # Numeric conversion attempts
        try:
            return int(value)  # Try integer first
        except ValueError:
            try:
                return float(value)  # Then try float
            except ValueError:
                return value  # Return original string if all conversions fail
        
    def onOtherConfigureEdited(self, item, column):
        """Slot function for handling other parameters editing"""
        if column == 1:
            new_value = item.text(1)
            key = item.text(0)
            if key in self.project.other_config:
                new_value = self.convert_value(new_value)
                self.project.other_config[key] = new_value
            else:
                raise ValueError(f"Invalid key {key} in other config")

    def onStartTrain(self):
        """Slot for starting training"""
        self.current_mode = "train"
        self._toggle_buttons("train", start=False)
        self.project.train()
        self._start_check_thread()

    def _start_check_thread(self):
        """Create a QTimer thread for checking training process"""
        self.check_thread = QTimer()
        self.check_thread.setInterval(1000)
        self.check_thread.timeout.connect(self._check_process)
        self.check_thread.start()

    def _stop_check_thread(self):
        """Remove the QTimer thread for checking training process"""
        if self.check_thread is not None:
            self.check_thread.stop()
            self.check_thread.deleteLater()
            self.check_thread = None

    def _check_process(self):
        """
        Check if the current process is still running.
        If not, perform corresponding actions based on the current mode.
        """
        if self.project.process.poll() is not None:
            if self.current_mode == "train":
                self._toggle_buttons("train", start=True)
            elif self.current_mode == "evaluate":
                self._toggle_buttons("evaluate", start=True)
            elif self.current_mode == "inference":
                self._toggle_buttons("inference", start=True)
            self.project.stop()
            self._stop_check_thread()

    def onEndTrain(self):
        """Slot for ending training"""
        self._toggle_buttons("train", start=True)
        self._stop_check_thread()
        self.project.stop()

    def onStartEvaluate(self):
        """Slot for starting evaluation"""
        self.current_mode = "evaluate"
        self._toggle_buttons("evaluate", start=False)
        self.project.evaluate()
        self._start_check_thread()

    def onEndEvaluate(self):
        """Slot for ending evaluation"""
        self._toggle_buttons("evaluate", start=True)
        self.project.stop()
        self._stop_check_thread()

    def onStartInference(self):
        """Slot for starting inference"""
        self.current_mode = "inference"
        self._toggle_buttons("inference", start=False)
        self.project.predict(inference_source=self.inference_source)
        self._start_check_thread()

    def onEndInference(self):
        """Slot for ending inference"""
        self._toggle_buttons("inference", start=True)
        self.project.stop()
        self._stop_check_thread()

    def _toggle_buttons(self, mode, start=True):
        """
        Toggle the enable state of start and end buttons based on mode.
        :param mode: 'train', 'evaluate', or 'inference'
        :param start: True to enable start button and disable end button, False otherwise
        """
        if mode == "train":
            self.StartTrain.setEnabled(start)
            self.EndTrain.setEnabled(not start)
        elif mode == "evaluate":
            self.StartEvaluate.setEnabled(start)
            self.EndEvaluate.setEnabled(not start)
        elif mode == "inference":
            self.StartInference.setEnabled(start)
            self.EndInference.setEnabled(not start)

        
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
        self.project.export()
    

class SplashScreen(QSplashScreen):
    """Custom splash screen with guaranteed initialization sequence"""
    def __init__(self, image_path):
        # Load image with size validation
        pixmap = self._validate_pixmap(image_path)
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        
        # Initialize UI components after base class setup
        self._init_ui_components(pixmap.size())
        
        # Force immediate display update
        self.finalize_initialization()

    def _validate_pixmap(self, path):
        """Ensure valid pixmap with fallback mechanism"""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            pixmap = QPixmap(800, 600)
            pixmap.fill(Qt.white)
        return pixmap

    def _init_ui_components(self, size):
        """Initialize progress bar and label with precise positioning"""
        # Progress bar (bottom 10% of splash)
        self.progress = QProgressBar(self)
        self.progress.setGeometry(
            int(size.width() * 0.1),       # x: 10% from left
            int(size.height() * 0.9),      # y: 85% from top
            int(size.width() * 0.8),       # width: 80% of total
            25                             # height
        )
        
        # Status label (above progress bar)
        self.status = QLabel("Initializing...", self)
        self.status.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font: bold 14px;
                background: transparent;
            }
        """)
        self.status.adjustSize()
        self.status.move(
            self.progress.x() + 10,        # Align with progress bar
            self.progress.y() - 35         # 35px above progress bar
        )

    def finalize_initialization(self):
        """Ensure complete rendering before first display"""
        self.setAttribute(Qt.WA_DontShowOnScreen, False)
        self.raise_()
        self.repaint()
        QApplication.processEvents()

def main():

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Initialize splash screen with guaranteed display
    splash = SplashScreen(WELCOME_PATH)
    splash.show()

    # Setup loading simulation
    progress = 0
    def simulate_loading():
        nonlocal progress
        progress += 2  # Increment by 2% each step
        splash.progress.setValue(progress)
        splash.status.setText(f"Loading modules... {progress}%")
        
        if progress >= 100:
            timer.stop()
            # Ensure complete splash cleanup
            splash.deleteLater()
            # Init main window AFTER splash destruction
            main_win = AnimalPoseTrackerPage()
            main_win.show()
    
    # Start loading simulation AFTER splash is fully shown
    timer = QTimer()
    timer.setInterval(20)  # 20ms interval for smooth progress
    timer.timeout.connect(simulate_loading)
    
    # Single-shot timer to start loading process
    QTimer.singleShot(100, timer.start)  # 100ms delay after display
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

        