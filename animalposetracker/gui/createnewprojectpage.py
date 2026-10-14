from PySide6.QtCore import Signal, Qt, QSignalBlocker
from PySide6.QtWidgets import  (QFileDialog, QMessageBox, QWidget, QMenu)
from PySide6.QtGui import QCursor, QIcon
from pathlib import Path

from .ui_createnewproject import Ui_CreateNewProject
from .utils import ListManager, CheckableListWidgetItem, ProjectFormBase
from animalposetracker.gui import LOGO_SMALL_PATH
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.cfg import MODEL_YAML_PATHS

class CreateNewProjectPage(ProjectFormBase, QWidget, Ui_CreateNewProject):
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

        # Set here rather than in the Designer file: the icon is a file on disk, and a path baked
        # into `ui_*.py` breaks as soon as the sources are regenerated.
        self.setWindowIcon(QIcon(LOGO_SMALL_PATH))
        self.project = AnimalPoseTrackerProject()
        self.project.local_path = Path.cwd()

        # Load project config
        self.SourceDataSelected.setTristate(True)
        self.SourceDataSelected.setEnabled(True)

        self.LocationPathDisplay.setReadOnly(True)
        self.LocationPathDisplay.setText(str(self.project.local_path))

        # Debounce timers for the name fields; see ProjectFormBase.
        self._init_project_form_timers()

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
                QMessageBox.warning(self, "Empty Directory", "No images found in selected folder")
            elif non_images:
                QMessageBox.warning(self, "Invalid Directory", 
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
        keypoints = self.KeypointConfig.value()
        classes = self.ClassConfig.value()
        project_config = {
            "project_path": str(self.project.project_path),
            "project_name": self.ProjectConfig.text(),
            "worker": self.WorkerConfig.text(),
            "model_type": self.ModelTypeSelection.currentText(),
            "model_scale": self.ModelScaleSelection.currentText(),
            "keypoints": keypoints,
            "visible": self.VisibleSelection.isChecked(),
            "keypoints_name": self.keypoint_manager.get_list_data(),
            "classes": classes,
            "classes_name": self.class_manager.get_list_data(),
        }
        # Both lists have to line up with their counts: the annotator and the dataset config read
        # them positionally, so a short list left the later keypoints unnamed.
        project_config["keypoints_name"] = self._fit_names(
            project_config["keypoints_name"], keypoints, "keypoint"
        )
        project_config["classes_name"] = self._fit_names(
            project_config["classes_name"], classes, "class"
        )

        return project_config

    @staticmethod
    def _fit_names(names, count, prefix):
        """Pad or trim a name list to exactly ``count`` entries."""
        names = list(names)[:count]
        names.extend(f"{prefix}{index + 1}" for index in range(len(names), count))
        return names

    def _validate_form(self, project_config):
        """Return a user-facing complaint, or None when the form can be submitted."""
        if not project_config["project_name"].strip():
            return "请填写项目名称。"
        if not project_config["worker"].strip():
            return "请填写 worker 名称。"
        if not self.LocationPathDisplay.text().strip():
            return "请选择项目保存位置。"
        if project_config["keypoints"] < 1:
            return "关键点数量至少为 1。"
        if project_config["classes"] < 1:
            return "类别数量至少为 1。"
        return None

    def onCreateProjectClicked(self):
        project_config = self.getAllProjectConfig()
        complaint = self._validate_form(project_config)
        if complaint is not None:
            QMessageBox.warning(self, "无法创建项目", complaint)
            return

        project_identity = {
            key: value for key, value in project_config.items()
            if key not in {"keypoints", "visible", "classes", "keypoints_name", "classes_name"}
        }
        self.project.update_config("project", project_identity)
        # The shape lives in dataset.yaml now (the project file keeps identity):
        # mirror the form's shape inputs there before the project is created.
        self._refresh_dataset_shape()
        self.project.local_path = Path(self.LocationPathDisplay.text())
        # Recorded before the attempt: rollback may only remove a directory this call created,
        # never a project that was already there (an existing one is refused, not overwritten).
        existed_before = (self.project.project_path / "project.yaml").exists()
        try:
            self.project.create_new_project()

            if self.CopySourceData.isChecked():
                sources = self.project.project_config['sources']
                if sources:
                    self.project.add_source_to_project(sources, 'copy')
                    # The copies are what the project should point at, and `create_new_project`
                    # already wrote the file -- so the new paths have to be persisted or the
                    # next load reverts `sources` to the originals.
                    self.project.save_configs("project")
        except Exception as error:
            # A half-created project on disk with no message was the old behaviour: the folder
            # stayed, the main window never advanced, and only a console traceback appeared.
            QMessageBox.critical(
                self, "创建项目失败",
                f"项目创建失败：\n{error}"
            )
            if not existed_before:
                self._rollback_project()
            return

        self.CreateProjectClicked.emit()

    def _rollback_project(self):
        """Remove the project directory this attempt created.

        Called only when no project was there beforehand, so an existing project can never be
        removed by a failed create.
        """
        import shutil

        path = self.project.project_path
        try:
            if path.is_dir() and (path / "project.yaml").exists():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
        
    def _refresh_dataset_shape(self):
        """Mirror the form's shape inputs onto the dataset config.

        The dataset file is the shape's single source now, so every shape
        edit (keypoint count, class count, name lists, visibility) lands
        there; flip_idx and the OKS sigmas follow the count.
        """
        keypoints_count = max(int(self.KeypointConfig.value()), 1)
        class_names = self._fit_names(
            self.class_manager.get_list_data(), int(self.ClassConfig.value()), "class"
        )
        keypoint_names = self._fit_names(
            self.keypoint_manager.get_list_data(), keypoints_count, "keypoint"
        )
        self.project.dataset_config.setdefault("keypoints_name", [])
        self.project.update_config("dataset", {
            "kpt_shape": [keypoints_count, 3 if self.VisibleSelection.isChecked() else 2],
            "flip_idx": list(range(keypoints_count)),
            "kpt_oks_sigmas": [1.0 / keypoints_count] * keypoints_count,
            "names": dict(enumerate(class_names)),
            "keypoints_name": keypoint_names,
        })

    def onKeypointConfigChanged(self, keypoints):
        self._refresh_dataset_shape()

    def onClassConfigChanged(self, classes):
        self._refresh_dataset_shape()

    def onKeypointChanged(self):
        """Handler for keypoint input change"""
        self._refresh_dataset_shape()

    def onClassChanged(self):
        """Handler for class input change"""
        self._refresh_dataset_shape()
    

        
        