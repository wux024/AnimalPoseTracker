from PySide6.QtCore import  Qt, QSettings, QFile, QTextStream, QTimer, QThread, Signal
from PySide6.QtWidgets import  (QMenu, QApplication, QFileDialog, QMainWindow,
                                 QMessageBox, QTreeWidget, QTreeWidgetItem, QLabel, 
                                 QTreeWidgetItemIterator, QSplashScreen, QProgressBar,
                                 QDialog)
from PySide6.QtGui import QCursor, QPixmap, QIcon, QColor
import os
import sys
from pathlib import Path
from collections import deque

from .ui_animalposetracker import Ui_AnimalPoseTracker
from .createnewprojectpage import CreateNewProjectPage
from .publicdatasetsprojectpage import PublicDatasetProjectPage
from .annotatorpage import AnnotatorPage
from .inferencerpage import InferencerPage
from .utils import (DatasetSplitDialog, split_dataset_sklearn, 
                    create_dataset_structure, copy_split_files,
                    create_coco_annotations, ConfigureValueDelegate)
from animalposetracker.gui import WindowFactory
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.utils import (convert_labels_to_coco, 
                                     convert_labels_to_yolo,
                                     KeyframeExtractor)
from animalposetracker.gui import (DARK_THEME_PATH, LIGHT_THEME_PATH, 
                                   LOGO_PATH_TRANSPARENT, LOGO_PATH, LOGO_SMALL_PATH,
                                   WELCOME_PATH)

# The parameter trees come from the Designer files, which mirror every knob the reference
# toolkits expose under those names. Only the ones this backend reads belong on screen: a row
# nothing consumes is a knob that silently does nothing, and a row whose key is missing from the
# project's runtime.yaml raises the moment it is edited.
_DROPPED_CONFIGURE_KEYS = {
    "TrainingConfigureEdit": frozenset({"overlap_mask", "mask_ratio", "dropout"}),
    "EvaluateConfigure": frozenset({
        "save_json", "save_hybrid", "save_txt", "save_conf", "save_crop", "augment",
    }),
    "InferenceConfigure": frozenset({
        "vid_stride", "stream_buffer", "visualize", "augment", "retina_masks", "embed",
        "stream", "show", "save_frame", "save_txt", "save_conf", "save_crop",
        "show_label", "font_size", "font", "pil", "masks", "probs", "filename",
        "color_mode", "txt_color",
    }),
}

# Rows the backend reads that the Designer file has no entry for, as ``(key, default)``. A dotted
# key addresses one field of a nested mapping in runtime.yaml (``tracking: {enabled: ...}``), which
# is the shape the prediction pipeline reads; the default seeds a project that predates the key.
_ADDED_CONFIGURE_KEYS = {
    "InferenceConfigure": (
        ("tracking.enabled", False),
        ("tracking.algorithm", "bytetrack"),
        ("pose_filter.method", "none"),
        ("pose_filter.window_length", 5),
        ("pose_filter.polyorder", 2),
        ("pose_filter.max_gap", 0),
    ),
}

# Rows whose value comes from a fixed set, so the value column offers them instead of asking for
# them to be typed. Booleans need no entry here: any row whose config value is a bool gets the
# same True/False dropdown. `cache` is listed because it takes two more words besides a bool.
_CONFIGURE_KEY_CHOICES = {
    "cache": ("True", "False", "ram", "disk"),
    "pose_filter.method": ("none", "median", "savgol"),
    "split": ("train", "val", "test"),
}

# Rows that look boolean in the config but also accept free text, so they keep the text editor:
# `pretrained` names a checkpoint when it is a string (`TrainingConfig.from_mapping`).
_FREETEXT_CONFIGURE_KEYS = frozenset({"pretrained"})

# Rows the backend does read, but under a different key than the Designer file uses.
_RENAMED_CONFIGURE_KEYS = {
    "InferenceConfigure": {"conf": "prediction_confidence", "iou": "prediction_iou"},
}


class _FrameExtractionThread(QThread):
    """Run the keyframe extractor off the GUI thread.

    Run inline, a longer extraction froze the whole window for its whole duration. The extractor
    has no stop hook, so this worker is only ever waited for briefly (see `closeEvent`).
    """

    failed = Signal(str)

    def __init__(self, extractor, video_paths, image_paths, mode, algorithm, n_clusters, interval,
                 parent=None):
        super().__init__(parent)
        self.extractor = extractor
        self.video_paths = list(video_paths)
        self.image_paths = list(image_paths)
        self.mode = mode
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.interval = interval

    def run(self):
        try:
            if self.video_paths:
                self.extractor.extract_keyframes_from_videos(
                    video_paths=self.video_paths,
                    mode=self.mode,
                    algorithm=self.algorithm,
                    n_clusters=self.n_clusters,
                    interval=self.interval,
                )
            for image_path in self.image_paths:
                self.extractor.extract_keyframes_from_images(
                    image_path=image_path,
                    mode=self.mode,
                    algorithm=self.algorithm,
                    n_clusters=self.n_clusters,
                    interval=self.interval,
                )
        except Exception as exc:  # the report goes to a dialog on the GUI thread via `failed`
            self.failed.emit(str(exc))


# Closing the window does not wait for a running extraction to finish (the extractor has no stop
# hook). Detached workers are kept here so the garbage collector does not destroy one mid-run.
_DETACHED_THREADS = []


def _detach_running_thread(thread):
    thread.setParent(None)
    _DETACHED_THREADS.append(thread)
    thread.finished.connect(
        lambda t=thread: _DETACHED_THREADS.remove(t) if t in _DETACHED_THREADS else None
    )


class AnimalPoseTrackerPage(QMainWindow, Ui_AnimalPoseTracker):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.initialize_controls()
        

    def initialize_controls(self):
        """Initialize the controls of the main page"""

        self.sub_page = None
        # Set here rather than in the Designer file: the icon and the logo are files on disk, and a
        # path baked into `ui_*.py` breaks the moment the sources are regenerated.
        self.setWindowIcon(QIcon(LOGO_SMALL_PATH))
        self.Logo.setPixmap(QPixmap(LOGO_PATH_TRANSPARENT))
        # Declared here, not only in _start_check_thread: End Evaluate/Inference read it, and on a
        # cold start neither had ever run, so the click raised AttributeError.
        self.check_thread = None
        # Which worker is running, or None. The buttons follow this rather than `Start.text()`.
        self.current_mode = None
        # set work directory
        os.chdir(Path.cwd())
        self.source_type = 'image' 
        self.project = AnimalPoseTrackerProject()
        self.extractor = KeyframeExtractor(Path.home())
        # The worker for one frame-extraction run; non-None only while it is running.
        self._extract_thread = None
        self.config_data = {}
        self.config_type = "project"
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
        self._shape_configure_trees()

        # Restore the theme the user last chose instead of always starting light.
        saved_theme = QSettings("Northeast Normal University", "AnimalPoseTracker").value("theme")
        self.onChangeTheme(saved_theme if saved_theme in ("dark", "light") else "light")
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

    def _open_sub_page(self, page_class):
        """Replace the current tool window, closing the previous one first.

        Every opener used to close the old window, build a new one, delete that one immediately
        and then build another -- a throwaway window per click.
        """
        if self.sub_page is not None:
            self.sub_page.close()
            self.sub_page.deleteLater()
        # The tool window is top-level, so it does not inherit this window's stylesheet; it is
        # handed the current theme instead of coming up in the default look.
        self.sub_page = WindowFactory.run(page_class, stylesheet=self.styleSheet())
        return self.sub_page

    def onCreateNewProject(self):
        """Slot for creating a new project"""
        self._open_sub_page(CreateNewProjectPage).CreateProjectClicked.connect(
            self._OpenConfigureTabPage
        )

    def onCreatePublicDatasetsProject(self):
        """Slot for opening public datasets"""
        self._open_sub_page(PublicDatasetProjectPage).CreateProjectClicked.connect(
            self._OpenConfigureTabPage
        )
    
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

    @staticmethod
    def _resolve_other_key(config, key):
        """Find where a tree key lives in the other config, following dots into nested mappings.

        Returns ``(container, leaf)``, or ``(None, None)`` when the config does not carry the key.
        The tree edits one scalar per row, so a dotted row addresses a single field of a nested
        mapping (``tracking.enabled``) rather than the mapping as a whole.
        """
        parts = key.split(".")
        container = config
        for part in parts[:-1]:
            if not isinstance(container, dict):
                return None, None
            container = container.get(part)
        if not isinstance(container, dict) or parts[-1] not in container:
            return None, None
        return container, parts[-1]

    @staticmethod
    def _ensure_other_key(config, key, default):
        """Make sure a tree key exists in the config so its row stays editable."""
        parts = key.split(".")
        container = config
        for part in parts[:-1]:
            nested = container.get(part)
            if not isinstance(nested, dict):
                nested = {}
                container[part] = nested
            container = nested
        container.setdefault(parts[-1], default)

    # Which runtime scope each parameter tree edits first (a row also falls
    # back to the train scope: batch / imgsz / fraction show up in several
    # trees but describe the training run).
    _RUNTIME_TREE_SCOPES = {
        "TrainingConfigureEdit": ("train",),
        "EvaluateConfigure": ("train",),
        "InferenceConfigure": ("predict",),
        "ExportConfigure": ("export",),
    }

    def _gui_runtime_tables(self):
        """Form-row spelling -> runtime.yaml path, per scope.

        Transition shim for this wave: the forms keep their historical row
        names (``workers``, ``imgsz``, ``conf`` ...) and each row maps onto
        the canonical runtime path it edits. The metadata-driven forms in
        the next wave replace this table wholesale.
        """
        cached = getattr(self, "_gui_runtime_tables_cache", None)
        if cached is not None:
            return cached

        from animalposetracker.config.schema import fields as schema_fields
        from animalposetracker.training.profiles import train_key_targets

        model_type = str(self.project.project_config.get("model_type") or "AnimalRTPose")
        targets = train_key_targets(model_type)

        train_table = {}
        for field in schema_fields(scope="train"):
            path = targets.get(field.key)
            if path is None:
                continue
            train_table.setdefault(field.key, path)
            for legacy in field.legacy:
                train_table.setdefault(legacy, path)

        tables = {"train": train_table, "predict": {}, "export": {}, "common": {}}
        for scope in ("predict", "export"):
            for field in schema_fields(scope=scope):
                key_name = f"{field.group}.{field.key}" if field.group else field.key
                path = f"{scope}.{key_name}"
                tables[scope].setdefault(key_name, path)
                tables[scope].setdefault(field.key, path)
                for legacy in field.legacy:
                    tables[scope].setdefault(legacy, path)
        for key in ("device", "project", "name", "exist_ok"):
            tables["common"][key] = f"common.{key}"

        self._gui_runtime_tables_cache = tables
        return tables

    def _runtime_path_for_gui_key(self, key, tree_name):
        """The runtime path one form row edits, or None (row shows as locked)."""
        tables = self._gui_runtime_tables()
        for scope in self._RUNTIME_TREE_SCOPES.get(tree_name, ()):
            if key in tables[scope]:
                return tables[scope][key]
        if key in tables["common"]:
            return tables["common"][key]
        if key in tables["train"]:
            return tables["train"][key]
        return None

    def _runtime_location(self, key, tree_name):
        """``(container, leaf)`` for a form row in the runtime tree."""
        path = self._runtime_path_for_gui_key(key, tree_name)
        if path is None:
            return None, None
        return self._resolve_other_key(self.project.runtime_config, path)

    def _runtime_location_any(self, key):
        """``(container, leaf)`` without a tree context (the choice delegate)."""
        tables = self._gui_runtime_tables()
        for scope in ("train", "predict", "export", "common"):
            if key in tables[scope]:
                return self._resolve_other_key(self.project.runtime_config, tables[scope][key])
        return None, None

    def configure_choices(self, key):
        """The fixed set of values a parameter row accepts, or ``()`` when it takes free text.

        The value delegate asks this per row, so a boolean or an algorithm name is picked from a
        list instead of typed -- a typo there used to reach the worker and fail far from the cell.
        """
        choices = _CONFIGURE_KEY_CHOICES.get(key)
        if choices:
            return tuple(choices)
        if key in _FREETEXT_CONFIGURE_KEYS:
            return ()
        if key == "tracking.algorithm":
            # Imported here rather than at module scope: the tracker package takes about a second
            # to import and the list is only needed once someone edits that row.
            from animalposetracker.tracking.config import ALGORITHMS

            return tuple(ALGORITHMS)
        container, leaf = self._runtime_location_any(key)
        if container is not None and isinstance(container[leaf], bool):
            return ("True", "False")
        return ()

    def _shape_configure_trees(self):
        """Make each parameter tree match what this backend reads: drop, rename and add rows.

        Runs at startup rather than being baked into the generated ``ui_*.py``: those files are
        rebuilt from the Designer sources, which would bring every dropped row straight back and
        lose every added one.
        """
        for tree_name, dropped in _DROPPED_CONFIGURE_KEYS.items():
            tree = getattr(self, tree_name)
            renamed = _RENAMED_CONFIGURE_KEYS.get(tree_name, {})
            for index in reversed(range(tree.topLevelItemCount())):
                item = tree.topLevelItem(index)
                key = item.text(0)
                if key in dropped:
                    tree.takeTopLevelItem(index)
                elif key in renamed:
                    item.setText(0, renamed[key])

        for tree_name, added in _ADDED_CONFIGURE_KEYS.items():
            tree = getattr(self, tree_name)
            for key, _default in added:
                item = QTreeWidgetItem()
                item.setText(0, key)
                tree.addTopLevelItem(item)

        # Kept on self: Qt does not take ownership of the delegate, and a collected one would
        # silently drop every dropdown back to a text editor.
        self._configure_delegate = ConfigureValueDelegate(self.configure_choices)
        for tree_name in (
            "TrainingConfigureEdit",
            "EvaluateConfigure",
            "InferenceConfigure",
            "ExportConfigure",
        ):
            getattr(self, tree_name).setItemDelegateForColumn(1, self._configure_delegate)

    def initialize_other_config(self):
        """Fill the parameter trees from the project's runtime configuration.

        Each row is looked up in the runtime tree through the path table (the
        historical row spelling maps onto the canonical runtime path). A row
        without a mapping -- a derived value like ``data``/``model``, or a
        key this project does not carry -- shows as locked instead of
        pretending to be editable.
        """
        trees = (
            "TrainingConfigureEdit",
            "EvaluateConfigure",
            "InferenceConfigure",
            "ExportConfigure",
        )
        for tree_name in trees:
            getattr(self, tree_name).blockSignals(True)
        for tree_name in trees:
            iterator = QTreeWidgetItemIterator(getattr(self, tree_name))
            while iterator.value():
                    item = iterator.value()
                    key = item.text(0)  # First column contains parameter names, dotted if nested
                    # Only update if the key exists in the runtime config
                    container, leaf = self._runtime_location(key, tree_name)
                    if container is not None:
                        item.setText(1, str(container[leaf]))
                        item.setFlags(item.flags() | Qt.ItemIsEditable)
                        item.setData(1, Qt.ForegroundRole, None)
                        item.setToolTip(1, "")
                    else:
                        # A row this project does not carry must not keep the value of the
                        # project opened before it: an edit to it would be silently discarded on
                        # save, while the stale value looked editable. Show it as unavailable.
                        item.setText(1, "")
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setData(1, Qt.ForegroundRole, QColor(150, 150, 150))
                        item.setToolTip(1, f"'{key}' is not present in this project's configuration")
                    iterator += 1 
        for tree_name in trees:
            getattr(self, tree_name).blockSignals(False)

    def loadRecentProjects(self):
        """Load saved projects from QSettings"""
        settings = QSettings("Northeast Normal University", "AnimalPoseTracker")
        try:
            projects = settings.value("recentProjects", [], type=list) or []
            # One damaged entry used to abort the whole constructor: `Path(None)` raises, and the
            # main window then never appeared, with no way for the user to repair the setting.
            self.recent_projects.extend(
                path for path in projects
                if isinstance(path, str) and Path(path).exists()
            )
        except (TypeError, ValueError):
            self.recent_projects.clear()

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
        # Flushed explicitly: a crash before the next event-loop turn otherwise lost the write.
        settings.sync()

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

    def closeEvent(self, event):
        """Close the window the same way the Exit action does.

        The title-bar X used to take the default handler, which left a running worker behind,
        dropped unsaved config edits and never closed the tool window.
        """
        if self.project is not None:
            self.project.save_configs()
        if self.project is not None and self.project.process is not None:
            # Orderly stop, the same contract the End button uses: the trainer writes its
            # checkpoint and exits with 130. `project.stop()` would kill it outright.
            self.project.request_training_stop()
            self.project.stop()
        self._stop_check_thread()
        if self._extract_thread is not None and self._extract_thread.isRunning():
            # The extractor has no stop hook: wait a moment only, then detach the worker, so
            # closing neither blocks on a long run nor lets Qt destroy a running thread.
            if not self._extract_thread.wait(3000):
                _detach_running_thread(self._extract_thread)
        if self.sub_page is not None:
            self.sub_page.close()
            self.sub_page = None
        event.accept()

    def onOpenAnnotator(self):
        """Slot for opening the annotator tool"""
        self._open_sub_page(AnnotatorPage)

    def onOpenInferencer(self):
        """Slot for opening the inferencer tool"""
        self._open_sub_page(InferencerPage)

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
            # An open tool window is top-level and keeps its own stylesheet, so it has to be told
            # about the change; otherwise the main window goes dark around a light tool window.
            if self.sub_page is not None:
                self.sub_page.setStyleSheet(self.styleSheet())
            # Remembered so the choice survives a restart; it used to reset to light every launch.
            QSettings("Northeast Normal University", "AnimalPoseTracker").setValue("theme", theme)
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
        config_type = Path(yaml_file).stem.lower()
        if config_type not in self.project.CONFIG_TYPES:
            # Returns rather than falling through: `config_data` used to stay at its previous
            # value (or `{}`), the tree was filled from it anyway, and Save then asked for a
            # `<type>_config` attribute that does not exist.
            QMessageBox.warning(
                self, 
                "Invalid Configuration Type", 
                "Please select a valid configuration type (project, dataset, model or runtime)"
            )
            return
        self.config_type = config_type
        self.config_data = getattr(self.project, f"{config_type}_config")
        self.DisplayConfigureFile()

    def DisplayConfigureFile(self):
        """Show the config in the tree without writing it back.

        Filling the tree sets each row's path through `setData`, and `setData` emits
        `itemChanged` -- the same signal the editor is connected to. Displaying a project used to
        run every value through `convert_value` as a side effect, so `'0012'` became `12`,
        `'1.0'` became `1.0` and `''` became `None`; saving after a plain look then wrote those
        types to disk and the project could not be loaded again.
        """
        self.ConfigureFile.clear()
        self.ConfigureFile.blockSignals(True)
        try:
            self.AddItemsToTree(self.config_data)
        finally:
            self.ConfigureFile.blockSignals(False)
    
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
            else:
                # A stray non-image file (a label, a `desktop.ini`) used to abort the whole
                # selection; the images are still perfectly usable, so it is only reported.
                self.project.add_source_to_project(dir_path)
                if non_images:
                    self.statusBar().showMessage(
                        f"{len(images)} images selected; ignored {len(non_images)} non-image file(s)"
                    )
                self.SelectionVideosImagesLabel.setText(f"{len(images)} images selected")
        
    def onClearVideosImages(self):
        """Slot for clearing videos/images selection"""
        self.SelectionVideosImagesLabel.setText("0 video or image selected")
        # A dict, not a set literal: `{"sources", []}` raised `TypeError: unhashable type:
        # 'list'` before it could clear anything, on every click.
        self.project.update_config("project", {"sources": []})

    def onExtractFrames(self):
        """Slot for extracting frames"""
        extract_mode = self.ExtractionMethodSelection.currentText()
        if extract_mode == "manual":
            # That path builds its own `QApplication` (utils/extract_frame.py), which cannot exist
            # beside the running one -- it raised RuntimeError inside the click. Labelling has its
            # own page, so point the user there instead.
            QMessageBox.information(
                self, "手动抽帧",
                "手动抽帧请在标注工具里进行：点击 “Start Label Frames”。"
            )
            return
        if self._extract_thread is not None and self._extract_thread.isRunning():
            QMessageBox.information(self, "Frame Extraction", "Frame extraction is already running.")
            return
        self.extractor.output_dir = self.project.project_path / "sources" / "extracted"
        extract_algorithm = self.ExtractionAlgorithmSelection.currentText()
        sources = self.project.project_config.get("sources", [])
        n_clusters = self.ClusterStepSetup.value()
        interval = self.SampleIntervalSetup.value()
        if not sources:
            QMessageBox.warning(self, "No Sources", "No sources found in project.yaml")
            return
        # Decided per source, not from the first one: a mixed selection used to have its later
        # entries read as whatever type the first entry happened to be.
        video_paths = [s for s in sources if not Path(s).is_dir()]
        image_paths = [s for s in sources if Path(s).is_dir()]
        self.source_type = "image" if image_paths and not video_paths else "video"
        # Off the GUI thread: run inline, a longer extraction froze the whole window with no way
        # to tell that anything was happening until it finished.
        self.statusBar().showMessage("Frame extraction running...")
        self.ExtracFrames.setEnabled(False)
        self._extract_thread = _FrameExtractionThread(
            self.extractor, video_paths, image_paths, extract_mode, extract_algorithm,
            n_clusters, interval, parent=self,
        )
        self._extract_thread.finished.connect(self._on_frame_extraction_finished)
        self._extract_thread.failed.connect(self._on_frame_extraction_failed)
        self._extract_thread.start()

    def _on_frame_extraction_finished(self):
        self.ExtracFrames.setEnabled(True)
        self.statusBar().showMessage("Frame extraction complete.")

    def _on_frame_extraction_failed(self, message):
        self.ExtracFrames.setEnabled(True)
        self.statusBar().showMessage("Frame extraction failed.")
        QMessageBox.warning(self, "Frame Extraction Failed", message)

    def onStartLabelFrames(self):
        """Slot for starting frame labeling"""
        self.onOpenAnnotator()

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
                convert_labels_to_coco(extracted_path, self.project.dataset_config)
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
        # Through the same guarded path as Start Train: resuming is the same worker, and this
        # used to disable the buttons by hand and never start the poll timer, so nothing brought
        # them back.
        self._start_worker("train", lambda: self.project.resume(resume_path))

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
                # EDIT MODE: Enable editing unless this project does not carry the key -- that
                # row stays locked, or an edit to it would be silently discarded on save.
                container, _leaf = self._runtime_location(
                    item.text(0), tree_widget.objectName()
                )
                if container is not None:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                # SAVE MODE: Disable editing and save changes
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            iterator += 1  # Move to next item
        
        if not is_edit_mode:
            # Save changes to config file
            self.project.save_configs("runtime")

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

        # An empty cell is left as the empty string. Mapping it to None wrote `null` into the
        # config for a field the user merely cleared, and `_generate_project_path` then failed
        # joining a non-string component.
        if not value:
            return ""
        if value in ("None", "Null", "null", "none"):
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
        """Slot function for handling runtime parameter editing"""
        if column == 1:
            new_value = item.text(1)
            key = item.text(0)
            tree = item.treeWidget()
            tree_name = tree.objectName() if tree is not None else ""
            container, leaf = self._runtime_location(key, tree_name)
            if container is not None:
                container[leaf] = self.convert_value(new_value)
            else:
                # A key the config does not carry would be dropped on save; say so instead of
                # raising, because the value the user just typed is still on screen.
                self.statusBar().showMessage(f"{key} is not a configuration key; the edit was ignored")

    def onStartTrain(self):
        """Slot for starting training"""
        self._start_worker("train", self.project.train)

    def _start_check_thread(self):
        """Start the one-second poll timer that watches the running worker."""
        if self.check_thread is not None:
            self.check_thread.stop()
            self.check_thread.deleteLater()
        self.check_thread = QTimer()
        self.check_thread.setInterval(1000)
        self.check_thread.timeout.connect(self._check_process)
        self.check_thread.start()

    def _stop_check_thread(self):
        """Stop the poll timer, if it was ever started."""
        if getattr(self, "check_thread", None) is not None:
            self.check_thread.stop()
            self.check_thread.deleteLater()
            self.check_thread = None

    def _check_process(self):
        """
        Check if the current process is still running.
        If not, perform corresponding actions based on the current mode.
        """
        if self.current_mode in {"train", "evaluate"}:
            self._display_project_events()

        process = self.project.process
        if process is None:
            # Nothing to watch. This used to raise on `None.poll()` *before* the timer was
            # stopped, so the timer survived and threw once a second for the rest of the session
            # while the Start button stayed greyed out.
            self._stop_check_thread()
            return

        if process.poll() is None:
            return

        returncode = process.returncode
        if self.current_mode == "train":
            self._toggle_buttons("train", start=True)
        elif self.current_mode == "evaluate":
            self._toggle_buttons("evaluate", start=True)
        elif self.current_mode == "inference":
            self._toggle_buttons("inference", start=True)
        self.project.stop()
        self._stop_check_thread()
        self._report_worker_exit(returncode)
        self.current_mode = None

    def _report_worker_exit(self, returncode):
        """Surface the training CLI's exit-code contract in the status bar.

        The worker documents 0 for a normal finish (early stopping included), 130 for an external
        stop request and 1 for an error, but nothing here read it -- a run that died before it
        opened its event stream reported nothing at all.
        """
        if returncode in (0, None):
            return
        if returncode == 130:
            self.statusBar().showMessage("训练已按请求停止，检查点已保存。")
            return
        self.statusBar().showMessage(
            f"训练异常退出（退出码 {returncode}）。详见项目 runs 目录下的训练日志。"
        )

    @staticmethod
    def _displayed_loss(metrics):
        """The loss worth showing, on the same scale the console uses.

        The SimCC head's ``loss`` is a batch sum that grows with batch size, so it cannot be read
        as a training curve; its per-image average is the comparable figure, exactly as the
        training console already reports it.
        """
        if "loss_simcc_per_sample" in metrics:
            return metrics["loss_simcc_per_sample"]
        return metrics.get("loss")

    @staticmethod
    def _metric_details(metrics) -> str:
        """The one-line metric summary both validation and evaluation events use.

        The validators fill one AP namespace or the other depending on the project's
        best_metric, and the SimCC validator reports no loss at all.
        """
        entries = (
            ("loss", AnimalPoseTrackerPage._displayed_loss(metrics), ".4g"),
            ("pose mAP50-95", metrics.get("coco/AP", metrics.get("ultralytics/AP")), ".3f"),
            ("pose mAP50", metrics.get("coco/AP50", metrics.get("ultralytics/AP50")), ".3f"),
            ("PCK", metrics.get("PCK"), ".3f"),
        )
        return ", ".join(
            f"{label}={value:{digits}}"
            for label, value, digits in entries
            if isinstance(value, (int, float))
        )

    def _display_project_events(self):
        """Show the latest structured worker status in the application status bar."""
        for event in self.project.read_training_events():
            kind = event.get("event")
            epoch = event.get("epoch")
            epochs = event.get("epochs")
            if kind == "batch":
                loss = self._displayed_loss(event.get("metrics", {}))
                detail = f", loss={loss:.4g}" if isinstance(loss, (int, float)) else ""
                self.statusBar().showMessage(
                    f"Training epoch {epoch}/{epochs}, batch {event.get('step')}/{event.get('steps')}{detail}"
                )
            elif kind == "validation":
                details = self._metric_details(event.get("metrics", {}))
                detail = f", {details}" if details else ""
                self.statusBar().showMessage(f"Validation epoch {epoch}/{epochs}{detail}")
            elif kind == "evaluation":
                # The evaluate worker runs once and exits; its metrics are the whole point of
                # the run, and they arrive as one event.
                details = self._metric_details(event.get("metrics", {}))
                summary = event.get("message") or "Evaluation complete."
                self.statusBar().showMessage(f"{summary}{', ' + details if details else ''}")
            elif kind == "pretrained":
                self.statusBar().showMessage(event.get("message", "Pretrained weights loaded."))
            elif kind == "stopped":
                self.statusBar().showMessage(
                    event.get("message") or "Training stopped; checkpoint saved."
                )
            elif kind == "failed":
                self.statusBar().showMessage(f"Training failed: {event.get('message', 'unknown error')}")
            elif kind == "finished":
                self.statusBar().showMessage("Training finished.")

    def onEndTrain(self):
        """Request an orderly stop after the custom trainer saves its checkpoint."""
        if self.project.process is None:
            # Disabling the button first meant a stray click left it permanently dead, because
            # nothing ever re-enabled it when no run was there to stop.
            self.statusBar().showMessage("当前没有正在进行的训练。")
            return
        self.EndTrain.setEnabled(False)
        self.project.request_training_stop()
        self.statusBar().showMessage("Stopping after the current training batch...")

    def _start_worker(self, mode, launch):
        """Run one worker, keeping the three tabs from starting over each other.

        `project.process` holds a single process, so starting an evaluate or inference while
        training was running replaced it: the trainer could no longer be stopped, its poll timer
        was gone, and the Start button for it never came back.
        """
        if self.project.process is not None and self.project.process.poll() is None:
            QMessageBox.warning(
                self, "已有任务在运行",
                f"当前正在运行「{self.current_mode or '未知'}」，请先结束它再开始新的任务。",
            )
            return
        self.current_mode = mode
        self._toggle_buttons(mode, start=False)
        try:
            launch()
        except Exception as error:
            # The backend raises for real reasons (a missing checkpoint, a bad config). Leaving
            # the buttons flipped left the tab unusable until the application was restarted.
            self._toggle_buttons(mode, start=True)
            self.current_mode = None
            QMessageBox.critical(self, "无法启动", f"任务启动失败：\n{error}")
            return
        self._start_check_thread()

    def onStartEvaluate(self):
        """Slot for starting evaluation"""
        self._start_worker("evaluate", self.project.evaluate)

    def onEndEvaluate(self):
        """Slot for ending evaluation"""
        self._toggle_buttons("evaluate", start=True)
        self.project.stop()
        self._stop_check_thread()
        self.current_mode = None

    def onStartInference(self):
        """Slot for starting inference"""
        self._start_worker(
            "inference",
            lambda: self.project.predict(inference_source=self.inference_source),
        )

    def onEndInference(self):
        """Slot for ending inference"""
        self._toggle_buttons("inference", start=True)
        self.project.stop()
        self._stop_check_thread()
        self.current_mode = None

    def _toggle_buttons(self, mode, start=True):
        """
        Toggle the enable state of start and end buttons based on mode.
        :param mode: 'train', 'evaluate', or 'inference'
        :param start: True to enable start button and disable end button, False otherwise
        """
        if mode == "train":
            self.StartTrain.setEnabled(start)
            self.EndTrain.setEnabled(not start)
            # Resume shares the training tab's single worker, so it follows the same switch;
            # leaving it out left it disabled forever after one resume.
            self.ResumeTrain.setEnabled(start)
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
        # Export and evaluation both resolve this through _resolve_project_weights, which only
        # accepts checkpoints -- offering .onnx here guaranteed a ValueError further down.
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select Model Weights",
            filter="PyTorch Weights (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.project.set_runtime_value("common.selected_weights", file_path)
            self.project.save_configs("runtime")

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

    # Qt 6 scales high-DPI displays by default; `AA_EnableHighDpiScaling` and
    # `AA_UseHighDpiPixmaps` are deprecated no-ops there, so only the rounding policy is set.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Initialize splash screen with guaranteed display
    splash = SplashScreen(WELCOME_PATH)
    splash.show()

    # Setup loading simulation
    progress = 0
    holder = {}
    def simulate_loading():
        nonlocal progress
        progress += 2  # Increment by 2% each step
        splash.progress.setValue(progress)
        splash.status.setText(f"Loading modules... {progress}%")
        
        if progress >= 100:
            timer.stop()
            # Ensure complete splash cleanup
            splash.deleteLater()
            # Init main window AFTER splash destruction. Held in `holder` rather than a local:
            # the closure would otherwise let the window be collected once it returns.
            holder["main_win"] = AnimalPoseTrackerPage()
            holder["main_win"].show()
    
    # Start loading simulation AFTER splash is fully shown
    timer = QTimer()
    timer.setInterval(20)  # 20ms interval for smooth progress
    timer.timeout.connect(simulate_loading)
    
    # Single-shot timer to start loading process
    QTimer.singleShot(100, timer.start)  # 100ms delay after display
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
