from PySide6.QtCore import Qt, QFile, QTextStream, QTimer, QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QTreeWidget, QMessageBox, QTreeWidgetItem, QTreeWidgetItemIterator, QGraphicsScene, QMenu
from PySide6.QtGui import QPixmap, QPainter, QIcon

import os
import sys
from pathlib import Path
import yaml
import json

from animalposetracker.gui import (DARK_THEME_PATH, LIGHT_THEME_PATH, COLORS, LOGO_SMALL_PATH)
from animalposetracker.utils.dataset import convert_labels_to_coco, convert_labels_to_yolo

from .ui_annotator import Ui_Annotator
from .utils import ZoomableGraphicsView, DrawingBoard, AnnotationTarget, AnnotationViewer

class AnnotatorPage(QMainWindow, Ui_Annotator):
    """Main application window for animal pose annotation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        self.initialize_controls()
        

    def initialize_controls(self):
        """Initialize UI controls and state"""
        os.chdir(Path.cwd())
        # Set here rather than in the ui module: the icon is a file on disk, and a path baked
        # into the module would break on any other machine.
        self.setWindowIcon(QIcon(LOGO_SMALL_PATH))
        self.project_config = {}
        self.images = {}
        self.config_path = None
        self.current_image_index = -1
        self.current_image_path = ""
        self.current_image = ""
        self.sorted_keys = []
        self.labels_cache = {}
        self._CreateGraphicsScene()
        self.ConfigureDisplay.header().close()
        # One flag per config type: the buttons are independent, and a single shared flag made
        # one editor's save write its data under another type's key.
        self._edit_modes = {'keypoints': False, 'classes': False, 'skeleton': False}
        self.current_drawing_mode = "none"

        self.EditKeypoints.setEnabled(False)
        self.EditSkeleton.setEnabled(False)
        self.EditClasses.setEnabled(False)

        self.actionDrawBBox.setEnabled(False)
        self.actionDrawLine.setEnabled(False)
        self.actionDrawPoint.setEnabled(False)
        self.actionBuildSkeleton.setEnabled(False)

        # No hard-coded theme: the main window applies its current stylesheet to this window when
        # it opens it, and forcing light here left a light tool window inside a dark application.
        saved_theme = QSettings("Jimei University", "AnimalPoseTracker").value("theme")
        self.onChangeTheme(saved_theme if saved_theme in ("dark", "light") else "light")

    def setupConnections(self):
        """Connect all UI signals to their corresponding slot functions"""
        # File operations
        self.actionOpenConfigureFile.triggered.connect(self.onOpenConfigFile)
        self.actionOpenFileFolder.triggered.connect(self.onOpenFileFolder)
        self.actionSaveLabel.triggered.connect(self.onSaveLabel)
        self.actionDeleteLabel.triggered.connect(self.onDeleteLabel)
        
        # Navigation controls
        self.actionNextFrame.triggered.connect(self.onNextFrame)
        self.actionPreviousFrame.triggered.connect(self.onPreviousFrame)
        
        # Annotation tools
        self.actionDrawBBox.triggered.connect(lambda: self.setDrawingMode("rect"))
        self.actionDrawPoint.triggered.connect(lambda: self.setDrawingMode("point"))
        self.actionDrawLine.triggered.connect(lambda: self.setDrawingMode("line"))
        self.actionBuildSkeleton.triggered.connect(lambda: self.setDrawingMode("bline"))
        self.actionNoLabel.triggered.connect(lambda: self.setDrawingMode("none"))
        
        # Configuration editing
        self.EditClasses.clicked.connect(self.onEditParameters)
        self.EditSkeleton.clicked.connect(self.onEditParameters)
        self.EditKeypoints.clicked.connect(self.onEditParameters)

        self.ConfigureDisplay.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ConfigureDisplay.customContextMenuRequested.connect(self._show_config_context_menu)
        
        # View settings
        self.actionDark.triggered.connect(lambda: self.onChangeTheme("dark"))
        self.actionLight.triggered.connect(lambda: self.onChangeTheme("light"))
        
        # Help
        self.actionHelp.triggered.connect(self.onShowHelp)

        # Export
        self.actionExportCOCOFormat.triggered.connect(lambda: self.onExportFormat("COCO"))
        self.actionExportYOLOFormat.triggered.connect(lambda: self.onExportFormat("YOLO"))
        
        self.Visible.addItems(['Invisible', 'Occluded', 'Visible'])
        self.Visible.setCurrentIndex(2)

        # Line Width and Radius
        self.LineWidth.valueChanged.connect(self.onLineWidthChanged)
        self.Radius.valueChanged.connect(self.onRadiusChanged)

    def _CreateGraphicsScene(self):
        """Create and configure the graphics scene"""
        self._ReplaceGraphicsView()
        self.scene = QGraphicsScene()
        self.FrameDisplay.setScene(self.scene)
        self._init_drawing_board()
        self.FrameDisplay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.FrameDisplay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.FrameDisplay.setRenderHints(
            QPainter.Antialiasing | 
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing
        )

    def _init_drawing_board(self):
        """Initialize the drawing board"""
        # The board is rebuilt for every image, and each one attached its own `AnnotationViewer`
        # (and a tree signal connection) to the shared label view. One viewer is reused instead,
        # so navigating a folder no longer accumulates them.
        if getattr(self, "labels_view", None) is None:
            self.labels_view = AnnotationViewer(self.LabelInformationView)

        self.drawing_label = DrawingBoard(keypoints=self.KeypointsSeletion,
                                          classes=self.ClassesSelection,
                                          skeletons=self.SkeletonSelection,
                                          labels=self.LabelInformationView,
                                          visible=self.Visible,
                                          labels_view=self.labels_view)
        
        if hasattr(self, 'LineWidth'):
            self.drawing_label.pen_settings["width"] = self.LineWidth.value()
        if hasattr(self, 'Radius'):
            self.drawing_label.pen_settings["radius"] = self.Radius.value()
        if hasattr(self.drawing_label, 'targets') and self.drawing_label.targets:
            self.drawing_label.set_current_target()
        self.drawing_label.setAutoFillBackground(True)
        self.drawing_label.setAttribute(Qt.WA_TranslucentBackground)
        self.drawing_label_proxy = self.scene.addWidget(self.drawing_label)
        self.drawing_label_proxy.setZValue(1)
        self.drawing_label.setVisible(True)
        self.drawing_label.addskeletons.connect(self.onAddSkeletons)
        

    def _ReplaceGraphicsView(self):
        """Replace default graphics view with custom zoomable version"""
        original_view = self.FrameDisplay
        self.FrameDisplay = ZoomableGraphicsView(self.FrameDisplayGroup)
        self.FrameDisplay.setObjectName(original_view.objectName())
        self.FrameDisplay.setStyleSheet(original_view.styleSheet())
        self.FrameDisplayGroupLayout.replaceWidget(original_view, self.FrameDisplay)
        original_view.deleteLater()

    def setDrawingMode(self, mode):
        """Set the current drawing mode and update UI"""
        valid_modes = ["none", "point", "rect", "line", "bline"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}")
        if mode == "none":
            self.actionDrawLine.setEnabled(False)
            self.actionDrawPoint.setEnabled(False)
            self.actionBuildSkeleton.setEnabled(False)
        elif mode == "rect":
            self.actionDrawPoint.setEnabled(True)
            self.actionDrawLine.setEnabled(False)
            self.actionBuildSkeleton.setEnabled(False)
        elif mode == "point":
            self.actionDrawLine.setEnabled(True)
            self.actionBuildSkeleton.setEnabled(True)
        
        self.current_drawing_mode = mode
        self.drawing_label.set_drawing_mode(mode)
        self.setFocus()

    def onOpenFileFolder(self):
        """Load images from selected directory"""
        folder_path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if not folder_path:
            return
        
        self.images.clear()
        self.labels_cache.clear()
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', 
                                '.gif', '.tif', '.tiff', '.webp'}
        folder = Path(folder_path)
        self.current_image_path = str(folder_path)
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in supported_extensions:
                self.images[img_file.name] = str(img_file)
        
        if not self.images:
            QMessageBox.warning(self, "Warning", "No supported images found")
            return
        
        self.statusBar().showMessage(f"Loaded {len(self.images)} images")
        self.sorted_keys = sorted(self.images.keys())
        self.current_image_index = 0
        self._DisplayCurrentImage()
        
        for img_name, img_path in self.images.items():
            json_path = Path(img_path).with_suffix('.json')
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        annotation_data = json.load(f)
                    self._LoadAnnotationtoCache(img_path, annotation_data)
                except Exception as e:
                    QMessageBox.warning(self, "Warning", 
                                    f"Failed to load annotation for {img_name}: {str(e)}")
        
        self.actionDrawBBox.setEnabled(True)
        self.actionDrawLine.setEnabled(False)
        self.actionDrawPoint.setEnabled(False)
        self.actionBuildSkeleton.setEnabled(False)
    
    def _LoadAnnotationtoCache(self, img_path, annotation_data):
        """Parse annotation data and store in labels_cache as AnnotationTarget objects
        
        Args:
            img_path: Path to the image file
            annotation_data: Parsed JSON annotation data
        """
        targets = []
        
        # Get image info (validate we have matching image in annotations)
        img_info = next((img for img in annotation_data.get('images', []) 
                        if img['file_name'] == Path(img_path).name), None)
        if not img_info:
            return
        
        # Process each annotation
        for i, ann in enumerate(annotation_data.get('annotations', [])):
            if ann['image_id'] != img_info['id']:
                continue
                
            # Create new target with unique color
            target = AnnotationTarget(
                target_id=ann.get('id', i),
                color=COLORS[i % len(COLORS)]
            )
            
            # 1. Process bounding box if exists. A point-only target has no box, and the file
            # records that as `null`; `len(None)` used to raise and the caller dropped *every*
            # annotation for that image, keypoints included.
            bbox = ann.get('bbox')
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y, w, h = bbox
                target.bounding_rect = {
                    'category_id': ann.get('category_id', 0),
                    'category_name': self._GetCategoryName(ann.get('category_id', 0)),
                    'bbox': [x, y, w, h],
                    'area': ann.get('area', w * h)
                }
            
            # 2. Process keypoints if they exist (COCO format: [x1,y1,v1, x2,y2,v2,...])
            if 'keypoints' in ann:
                kpt_data = ann['keypoints']
                for kpt_id in range(len(kpt_data) // 3):
                    idx = kpt_id * 3
                    x, y, v = kpt_data[idx], kpt_data[idx+1], kpt_data[idx+2]
                    
                    # Get keypoint name from config if available
                    kpt_name = (self.project_config['keypoint_names'][kpt_id] 
                            if ('keypoint_names' in self.project_config and 
                                kpt_id < len(self.project_config['keypoint_names']))
                            else f"kpt_{kpt_id}")
                    
                    target.keypoints[kpt_id] = {
                        'name': kpt_name,
                        'pos': [x, y, v]  # x, y, visibility
                    }
                    target.current_keypoint_id = kpt_id  # Track last used ID
            
            # 3. Process skeleton connections if they exist
            if 'skeleton' in self.project_config:
                for conn_id, (src_kpt, dst_kpt) in enumerate(self.project_config['skeleton']):
                    # Only create connection if both keypoints exist
                    if (src_kpt in target.keypoints and 
                        dst_kpt in target.keypoints):
                        src_pos = target.keypoints[src_kpt]['pos'][:2]  # Get x,y only
                        dst_pos = target.keypoints[dst_kpt]['pos'][:2]
                        
                        target.skeletons[conn_id] = {
                            'skeleton': [src_kpt, dst_kpt],
                            'pos': [*src_pos, *dst_pos]  # x1,y1,x2,y2
                        }
            
            targets.append(target)
        
        # Store in cache
        self.labels_cache[img_path] = {
            'targets': targets
        }
    
        # Update the DrawingBoard if this is the current image
        if img_path == self.current_image:
            self.drawing_label.targets = targets.copy()
            self.drawing_label.set_current_target(0)
            self.drawing_label._commit_changes()
    
    def _GetCategoryName(self, category_id):
        """Get category name from project config"""
        if ('classes_name' in self.project_config and 
            0 <= category_id < len(self.project_config['classes_name'])):
            return self.project_config['classes_name'][category_id]
        return f"class_{category_id}"

    def _DisplayCurrentImage(self):
        """Display current image with annotations"""
        if not self.images:
            self.scene.clear()
            return

        key = self.sorted_keys[self.current_image_index]
        image_path = Path(self.images[key])
        
        try:
            if hasattr(self, 'drawing_label') and self.drawing_label and self.current_image:
                self.labels_cache[self.current_image] = {
                    'targets': self.drawing_label.targets.copy()
                }


            pixmap = QPixmap(str(image_path))

            if pixmap.isNull():
                raise ValueError("Failed to load image")
            
            # Clear and setup scene
            self.scene.clear()
            self.scene.addPixmap(pixmap)

            self._init_drawing_board()
            
            # Update DrawingBoard dimensions
            self.drawing_label.setFixedSize(pixmap.size())
            self.drawing_label.main_pixmap = QPixmap(pixmap.size())
            self.drawing_label.main_pixmap.fill(Qt.transparent)
            self.drawing_label.temp_pixmap = QPixmap(pixmap.size())
            self.drawing_label.temp_pixmap.fill(Qt.transparent)

            self.current_image = str(image_path)
            if self.current_image in self.labels_cache:
                # A copy, not the cached list itself: the board clears and rebuilds its target
                # list, and sharing the object meant `board.targets.clear()` emptied the cache too.
                self.drawing_label.targets = list(
                    self.labels_cache[self.current_image]['targets']
                )
                self.drawing_label._commit_changes()
            
            # Set view parameters
            self.FrameDisplay.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.statusBar().showMessage(f"Displaying: {key}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")

    # Additional methods (unchanged from original)
    @staticmethod
    def _normalize_shape_config(config):
        """Expose the three edited lists regardless of the file's spelling.

        The shape's home is dataset.yaml now (``names`` as a mapping); this
        page still edits ``classes_name`` (historical) and ``keypoint_names``
        / ``skeleton`` lists, so it accepts the old ``keypoints_name``
        spelling on the way in and writes only the canonical one.
        """
        normalized = dict(config or {})
        class_names = normalized.get("classes_name")
        if not class_names:
            names = normalized.get("names")
            if isinstance(names, dict):
                class_names = [str(value) for _, value in sorted(names.items())]
            elif isinstance(names, (list, tuple)):
                class_names = [str(value) for value in names]
        normalized["classes_name"] = list(class_names or [])
        if not normalized.get("keypoint_names"):
            # `keypoints_name` is the old spelling; read it, never keep it.
            legacy = normalized.get("keypoints_name")
            shape = normalized.get("kpt_shape") or [0]
            normalized["keypoint_names"] = list(legacy) if legacy else [
                f"kpt_{index}" for index in range(int(shape[0]))
            ]
        normalized.pop("keypoints_name", None)
        normalized.setdefault("skeleton", [])
        return normalized

    def onOpenConfigFile(self):
        """Handle opening configuration file with safety checks"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Config File", "", 
            "Config Files (*.yaml *.yml)"
        )
        if not file_path:
            return
            
        try:
            self.config_path = file_path
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use safe loader instead of FullLoader
                self.project_config = self._normalize_shape_config(yaml.safe_load(f))
            
            self.EditKeypoints.setEnabled(True)
            self.EditSkeleton.setEnabled(True)
            self.EditClasses.setEnabled(True)
            
            # Validate required sections
            required = ['keypoint_names', 'classes_name', 'skeleton']
            if not all(k in self.project_config for k in required):
                raise ValueError("Missing required config sections")
                
            self._populate_comboboxes()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config file: {e}")

    def _populate_comboboxes(self):
        """Populate UI elements from config"""
        self.KeypointsSeletion.clear()
        self.ClassesSelection.clear()
        self.SkeletonSelection.clear()
        
        self.KeypointsSeletion.addItems(
            self.project_config['keypoint_names'])
        self.ClassesSelection.addItems(
            self.project_config['classes_name'])
        
        skeleton_links = [
            f"{src}->{dest}" for src, dest in self.project_config['skeleton']
        ]
        self.SkeletonSelection.addItems(skeleton_links)
        
    def _advance_image_index(self, step):
        """Safely advance image index with wrap-around handling."""
        if not self.images:
            return
        self.current_image_index += step
        self.current_image_index = self.current_image_index % len(self.images)
        self._DisplayCurrentImage()

    def onNextFrame(self):
        """Display the next image (with wrap-around)"""
        if not self._check_images_loaded():
            return
        self._advance_image_index(1)
        self.setDrawingMode("none")

    def onPreviousFrame(self):
        """Display the previous image (with wrap-around)"""
        if not self._check_images_loaded():
            return
        self._advance_image_index(-1)
        self.setDrawingMode("none")

    def _check_images_loaded(self):
        """Helper method to check if images are loaded"""
        if not self.images:
            self.statusBar().showMessage("No images loaded")
            return False
        return True
    
    def _show_config_context_menu(self, pos):
        """Show context menu with add/delete options for config table"""
        item = self.ConfigureDisplay.itemAt(pos)
        menu = QMenu()
        
        # Add "Add Row Above" and "Add Row Below" actions
        add_above_action = menu.addAction("Add Row Above")
        add_above_action.triggered.connect(lambda: self._add_config_row(pos, "above"))
        add_below_action = menu.addAction("Add Row Below") 
        add_below_action.triggered.connect(lambda: self._add_config_row(pos, "below"))
        
        # Add separator
        menu.addSeparator()
        
        # Add "Delete Row" action if clicking on existing row
        if item:
            delete_action = menu.addAction("Delete Row")
            delete_action.triggered.connect(lambda: self._delete_config_row(item))
        
        menu.exec_(self.ConfigureDisplay.viewport().mapToGlobal(pos))
    
    def _add_config_row(self, pos, insert_position="below"):
        """Add a new row to the configuration table with reliable editing"""
        item = self.ConfigureDisplay.itemAt(pos)
        config_type = self._get_current_config_type()
        
        if not config_type:
            QMessageBox.warning(self, "Warning", "Please enter edit mode first")
            return
        
        # Block signals during the operation
        self.ConfigureDisplay.blockSignals(True)
        
        try:
            # Create new item
            new_item = QTreeWidgetItem()
            new_item.setFlags(new_item.flags() | Qt.ItemIsEditable)
            
            # Set default values based on config type
            if config_type == 'classes':
                default_text = "New Class"
            elif config_type == 'keypoints':
                default_text = "New Keypoint"
            elif config_type == 'skeleton':
                # Generate default skeleton connection
                last_id = 0
                if self.ConfigureDisplay.topLevelItemCount() > 0:
                    last_item = self.ConfigureDisplay.topLevelItem(self.ConfigureDisplay.topLevelItemCount()-1)
                    try:
                        last_part = last_item.text(1).split('->')[-1]
                        last_id = int(last_part.strip())
                    except (ValueError, IndexError):
                        pass
                default_text = f"{last_id+1}->{last_id+2}"
            
            # Set initial text
            new_item.setText(0, str(self.ConfigureDisplay.topLevelItemCount()))
            new_item.setText(1, default_text)
            
            # Determine insert position
            if item:
                parent = self.ConfigureDisplay.invisibleRootItem()
                index = parent.indexOfChild(item)
                if insert_position == "above":
                    parent.insertChild(index, new_item)
                else:
                    parent.insertChild(index + 1, new_item)
            else:
                self.ConfigureDisplay.addTopLevelItem(new_item)
            
            # Renumber all rows
            self._renumber_rows()
            
            # Ensure proper editing triggers are set
            self.ConfigureDisplay.setEditTriggers(QTreeWidget.DoubleClicked | QTreeWidget.EditKeyPressed)
            
            # Make sure the item is visible and selected
            self.ConfigureDisplay.scrollToItem(new_item)
            self.ConfigureDisplay.setCurrentItem(new_item)
            
            # Start editing with a slight delay to ensure UI readiness
            QTimer.singleShot(100, lambda: (
                self.ConfigureDisplay.editItem(new_item, 1),
                self.ConfigureDisplay.setFocus()
            ))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add row: {str(e)}")
        finally:
            # Restore signal handling
            self.ConfigureDisplay.blockSignals(False)

    def _delete_config_row(self, item):
        """Delete row with validation for skeleton integrity"""
        config_type = self._get_current_config_type()
        
        if not config_type:
            QMessageBox.warning(self, "Warning", "Please enter edit mode first")
            return
        
        # Special validation for skeleton connections
        if config_type == 'skeleton':
            connection = item.text(1)
            src, dest = map(int, connection.split('->'))
            
            # Check if this connection is used in annotations
            if self._is_skeleton_used(src, dest):
                QMessageBox.warning(self, "Warning", 
                                "This connection is used in existing annotations!\n"
                                "Please delete annotations first.")
                return

        # Deleting a keypoint shifts every later keypoint's index, and the saved labels still
        # carry the old numbering -- so the annotations would quietly come to mean different
        # keypoints. Refused rather than silently remapped.
        if config_type == 'keypoints':
            index = self.ConfigureDisplay.indexOfTopLevelItem(item)
            if not self._keypoint_row_is_removable(index):
                QMessageBox.warning(
                    self, "Warning",
                    f"Keypoints that come after {item.text(1)} (index {index}) are still referenced.\n"
                    "Deleting it shifts the indices of the keypoints that follow, so existing "
                    "annotations and skeleton links would point at other keypoints.\n"
                    "Delete the related annotations and skeleton links first."
                )
                return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, 'Confirm', 
            f"Delete {item.text(1)}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
        
        # Perform deletion
        self.ConfigureDisplay.takeTopLevelItem(self.ConfigureDisplay.indexOfTopLevelItem(item))
        self._renumber_rows()
    
    def _get_current_config_type(self):
        """Determine which config type is currently being edited"""
        if self.EditClasses.text().startswith('Save'):
            return 'classes'
        elif self.EditKeypoints.text().startswith('Save'):
            return 'keypoints'
        elif self.EditSkeleton.text().startswith('Save'):
            return 'skeleton'
        return None

    def _renumber_rows(self):
        """Renumber all rows sequentially after add/delete operations"""
        for i in range(self.ConfigureDisplay.topLevelItemCount()):
            self.ConfigureDisplay.topLevelItem(i).setText(0, str(i))

    def _is_skeleton_used(self, src, dest):
        """Check if skeleton connection exists in any annotation"""
        for img_data in self.labels_cache.values():
            for target in img_data['targets']:
                for skeleton in target.skeletons.values():
                    if {src, dest} == set(skeleton['skeleton']):
                        return True
        return False

    def _referenced_keypoint_indices(self):
        """Every keypoint index that a cached annotation or a configured edge refers to."""
        indices = set()
        for img_data in self.labels_cache.values():
            for target in img_data['targets']:
                indices.update(target.keypoints)
                for skeleton in target.skeletons.values():
                    indices.update(skeleton['skeleton'])
        for edge in self.project_config.get('skeleton', []) or []:
            indices.update(edge)
        return indices

    def _keypoint_row_is_removable(self, index):
        """Whether removing this keypoint row leaves every remaining index meaning the same thing.

        Removing row ``i`` shifts every later keypoint down one, and nothing rewrites `skeleton`
        or the saved annotations -- so a referenced index at or after ``i`` would silently come to
        mean a different keypoint. Only rows past every reference are safe.
        """
        return all(referenced < index for referenced in self._referenced_keypoint_indices())

    def onEditParameters(self):
        """Toggle edit mode for the button that was clicked.

        Each button tracks its own state. One shared flag meant "Edit Classes" followed by
        "Edit Keypoints" saved the second button's type against the first button's data -- the
        class list was written into the keypoint list, and a skeleton list into the class list.
        """
        button = self.sender()
        button_map = {
            self.EditKeypoints: 'keypoints',
            self.EditClasses: 'classes',
            self.EditSkeleton: 'skeleton',
        }

        config_type = button_map.get(button)
        if config_type is None:
            return

        if not hasattr(self, '_edit_modes'):
            self._edit_modes = {}

        if not self._edit_modes.get(config_type):
            # Opening one editor closes any other that was left open, so the table always shows
            # the type the buttons claim it shows.
            for other, is_open in list(self._edit_modes.items()):
                if is_open and other != config_type:
                    self._edit_modes[other] = False
                    self._edit_button(other).setText(f"Edit {other.title()}")
            self._edit_modes[config_type] = True
            self._HandleConfigDisplay(config_type)
            button.setText(f"Save {config_type.title()}")
            self.ConfigureDisplay.itemChanged.connect(
                self.onConfigureEdited,
                Qt.QueuedConnection)
        else:
            self._edit_modes[config_type] = False
            self.ConfigureDisplay.blockSignals(True)
            try:
                self._SaveConfigChanges(config_type)
                button.setText(f"Edit {config_type.title()}")
                self.ConfigureDisplay.itemChanged.disconnect()
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))
            finally:
                self.ConfigureDisplay.blockSignals(False)

    def _edit_button(self, config_type):
        """The button that toggles one config type's editor."""
        return {
            'keypoints': self.EditKeypoints,
            'classes': self.EditClasses,
            'skeleton': self.EditSkeleton,
        }[config_type]

    def _SaveConfigChanges(self, config_type):
        """Validate and save configuration changes"""
        try:
            updated_data = []
            
            iterator = QTreeWidgetItemIterator(self.ConfigureDisplay)
            while iterator.value():
                item = iterator.value()
                if config_type == 'skeleton':
                    try:
                        src, dest = item.text(1).split('->')
                        updated_data.append([int(src.strip()), int(dest.strip())])
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid skeleton format: {item.text(1)}") from e
                else:
                    updated_data.append(item.text(1))
                
                iterator +=1
            
            # Update config and refresh UI
            if config_type == 'classes':
                self.project_config['classes_name'] = updated_data
            elif config_type == 'keypoints':
                self.project_config['keypoint_names'] = updated_data
            elif config_type =='skeleton':
                self.project_config['skeleton'] = updated_data
            
            document = dict(self.project_config)
            if "kpt_shape" in document:
                # A dataset.yaml keeps its own spelling: the class names live
                # there as a mapping, not as this page's editing list.
                document["names"] = dict(enumerate(document.get("classes_name") or []))
                document.pop("classes_name", None)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(document, 
                        f, 
                        indent=2, 
                        sort_keys=False, 
                        default_flow_style=False
                    )
            self._populate_comboboxes()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
            raise
    
    def _HandleConfigDisplay(self, config_type):
        """Generic handler for configuration display (DRY)"""
        config_map = {
            'classes': ('classes_name', ['Class ID', 'Class Name']),
            'keypoints': ('keypoint_names', ['Keypoint ID', 'Keypoint Name']),
            'skeleton': ('skeleton', ['Link ID', 'Connection'])
        }
        
        config_key, headers = config_map[config_type]
        data = self.project_config.get(config_key, [])
        
        self.ConfigureDisplay.clear()
        self.ConfigureDisplay.setHeaderLabels(headers)
        
        for idx, item in enumerate(data):
            list_item = QTreeWidgetItem(self.ConfigureDisplay)
            list_item.setText(0, str(idx))
            
            if config_type == 'skeleton':
            # Ensure item is in the correct format
                if isinstance(item, list) and len(item) == 2:
                    connection_text = f"{item[0]}->{item[1]}"
                else:
                    connection_text = "0->1"  # Default if invalid
                list_item.setText(1, connection_text)
            else:
                list_item.setText(1, str(item))
            
            list_item.setFlags(list_item.flags() | Qt.ItemIsEditable)
            list_item.setData(0, Qt.UserRole, idx)
        
        self.ConfigureDisplay.setEditTriggers(QTreeWidget.DoubleClicked | QTreeWidget.EditKeyPressed)


    def onConfigureEdited(self, item, column):
        """Slot function for handling other parameters editing"""
        if not hasattr(self, '_edit_lock'):
            self._edit_lock = False
        if self._edit_lock:
            return

        try:
            self._edit_lock = True
            if column == 1:
                new_value = item.text(1)
                # The row number, not the text of column 0: `text(0)` is a *string*, and using it
                # as a list index raised TypeError on every edit, which Qt then swallowed.
                row = int(item.text(0))
                if self.EditClasses.text().startswith('Save'):
                    if row < len(self.project_config.get('classes_name', [])):
                        self.project_config['classes_name'][row] = new_value
                elif self.EditKeypoints.text().startswith('Save'):
                    if row < len(self.project_config.get('keypoint_names', [])):
                        self.project_config['keypoint_names'][row] = new_value
                elif self.EditSkeleton.text().startswith('Save'):
                    if row < len(self.project_config.get('skeleton', [])):
                        self.project_config['skeleton'][row] = [
                            int(part.strip()) for part in new_value.split('->')
                        ]
        except (TypeError, ValueError):
            # A partially typed edge ("2-") is not a pair yet; the save button re-validates.
            pass
        finally:
            self._edit_lock = False

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
        except Exception as e:
            QMessageBox.warning(self, "Theme Error", str(e))

    def onShowHelp(self):
        """
        Handle showing help information.
        
        Displays application help documentation.
        """
        print("Showing help")
    
    def onSaveLabel(self):
        """Save current annotations to file.

        An empty target list is written too, as an annotation file with no annotations: that is
        how a deletion is persisted. Refusing to save it meant "delete all, then save" left the
        old boxes on disk, and reopening the folder brought them back.
        """
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "No image is open")
            return

        self.labels_cache[self.current_image] = {
            'targets': self.drawing_label.targets.copy()
        }
    
        image = {
                "id": self.current_image_index,
                "file_name": self.sorted_keys[self.current_image_index],
                "width": self.drawing_label.main_pixmap.width(),
                "height": self.drawing_label.main_pixmap.height(),
        }
        annotations = []
        for target in self.drawing_label.targets:
            if target.keypoints:
                keypoints = [
                        coord 
                        for kp_id in sorted(target.keypoints.keys())
                        for coord in target.keypoints[kp_id]["pos"]
                    ]
            else:
                keypoints = []
            bbox = target.bounding_rect.get('bbox')
            annotation = {
                "id": target.id,
                "image_id": self.current_image_index,
                "category_id": target.bounding_rect.get('category_id'),
                # `[]` rather than `null` for a point-only target: the loader reads a box back
                # with `len()`, and `null` made the file unreadable by this same page.
                "bbox": list(bbox) if bbox else [],
                "area": target.bounding_rect.get('area'),
                "iscrowd": int(self.IsCrowd.isChecked()),
                "keypoints": [*keypoints], 
            }
            annotations.append(annotation)
        coco_dict = {
            "images": [image],
            "annotations": annotations,
        }
        key = self.sorted_keys[self.current_image_index]
        label_file = Path(self.images[key]).with_suffix('.json')
        self._write_json_atomically(label_file, coco_dict)
        self.statusBar().showMessage(f"Saved {len(annotations)} annotations to {label_file.name}")

    @staticmethod
    def _write_json_atomically(path, payload):
        """Write JSON through a temporary file so a crash cannot leave a half-written label."""
        temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=4)
        os.replace(temporary, path)
    
    def onDeleteLabel(self):
        """Delete current annotations and persist the deletion."""
        if not self.drawing_label.targets:
            QMessageBox.warning(self, "Warning", "No annotations to delete")
            return
        self.drawing_label.targets = []
        self.drawing_label._commit_changes()
        # Written straight away: the button means "this image now has no annotations", and
        # leaving it in memory only made the deletion look like it had not happened.
        if self.current_image:
            self.onSaveLabel()

    def onLineWidthChanged(self, value):
        """Handle line width change"""
        if hasattr(self, 'drawing_label'):
            self.drawing_label.pen_settings['width'] = value
            self.drawing_label._commit_changes()
    
    def onRadiusChanged(self, value):
        """Handle radius change"""
        if hasattr(self, 'drawing_label'):
            self.drawing_label.pen_settings['radius'] = value
            self.drawing_label._commit_changes()
    
    def onAddSkeletons(self, value):
        """Handle skeleton link change"""
        if not hasattr(self, 'drawing_label'):
            return
        if not self.config_path:
            # Drawing without a config open used to raise `KeyError`/`TypeError` on `None`.
            QMessageBox.warning(
                self, "Warning",
                "Open the project configuration file (Open Config File) before editing skeleton links."
            )
            return

        self.SkeletonSelection.addItem(f"{value[0]}->{value[1]}")
        skeleton = self.project_config.setdefault('skeleton', [])
        if list(value) not in [list(edge) for edge in skeleton]:
            # Duplicates used to multiply: the same connection appended again and again, and the
            # combo grew one entry per redraw.
            skeleton.append(list(value))
        # Written atomically, like the labels.
        temporary = Path(self.config_path).with_suffix(
            Path(self.config_path).suffix + ".tmp"
        )
        with open(temporary, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.project_config,
                stream,
                indent=2,
                sort_keys=False,
                default_flow_style=False
            )
        os.replace(temporary, self.config_path)
    
    def onExportFormat(self, format='COCO'):
        """Handle export format change"""
        try:
            if format == 'COCO':
                output_dir = convert_labels_to_coco(
                    labels_path=self.current_image_path,  
                    project_config=self.project_config
                )
            elif format == 'YOLO':
                output_dir = convert_labels_to_yolo(
                    labels_path=self.current_image_path
                )
            else:
                QMessageBox.warning(self, "Warning", "Invalid export format")
                return
                
            self.statusBar().showMessage(f"Labels exported to {output_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            
        
    def closeEvent(self, event):
        """Offer to keep unsaved work before the window goes away.

        Closing used to clear the scene and accept, so every annotation not yet saved was lost
        with nothing to show for it.
        """
        if self._has_unsaved_work():
            answer = QMessageBox.question(
                self,
                "Unsaved annotations",
                "This image's annotations have not been saved. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                self.onSaveLabel()

        self.scene.clear()
        self.drawing_label = None
        event.accept()

    def _has_unsaved_work(self):
        """Whether the current image's labels differ from what is on disk."""
        if not self.current_image or not self.drawing_label:
            return False
        label_file = Path(self.images[self.sorted_keys[self.current_image_index]]).with_suffix(".json")
        if not label_file.exists():
            return bool(self.drawing_label.targets)
        try:
            saved = json.loads(label_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return len(saved.get("annotations", [])) != len(self.drawing_label.targets)

    def resizeEvent(self, event):
        """Maintain aspect ratio when resizing window"""
        super().resizeEvent(event)
        if self.scene.items():
            self.FrameDisplay.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

def main():
    app = QApplication(sys.argv)
    window = AnnotatorPage()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()