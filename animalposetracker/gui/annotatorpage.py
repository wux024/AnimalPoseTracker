from PySide6.QtCore import Qt, QFile, QTextStream, QTimer
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow, QTreeWidget,
                               QMessageBox, QTreeWidgetItem, QTreeWidgetItemIterator,
                               QGraphicsScene, QGraphicsView, QMenu)
from PySide6.QtGui import QPixmap, QPainter

import os
import sys
from pathlib import Path
import yaml
import json

from animalposetracker.gui import (DARK_THEME_PATH, LIGHT_THEME_PATH, COLORS)
from animalposetracker.utils.dataset import convert_labels_to_coco, convert_labels_to_yolo

from .ui_annotator import Ui_Annotator
from .utils import ZoomableGraphicsView, DrawingBoard, AnnotationTarget

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
        self._edit_mode = False
        self.current_drawing_mode = "none"

        self.EditKeypoints.setEnabled(False)
        self.EditSkeleton.setEnabled(False)
        self.EditClasses.setEnabled(False)

        self.actionDrawBBox.setEnabled(False)
        self.actionDrawLine.setEnabled(False)
        self.actionDrawPoint.setEnabled(False)
        self.actionBuildSkeleton.setEnabled(False)

        self.onChangeTheme("light")

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
        self.drawing_label = DrawingBoard(keypoints=self.KeypointsSeletion,
                                          classes=self.ClassesSelection,
                                          skeletons=self.SkeletonSelection,
                                          labels=self.LabelInformationView,
                                          visible=self.Visible)
        
        if hasattr(self, 'LineWidth'):
            self.drawing_label.pen_settings["width"] = self.LineWidth.value()
        if hasattr(self, 'Radius'):
            self.drawing_label.pen_settings["radius"] = self.Radius.value()
        if hasattr(self.drawing_label, 'targets') and self.drawing_label.targets:
            self.drawing_label.set_current_target()
        self.drawing_label.setAutoFillBackground(True)
        self.drawing_label.setAttribute(Qt.WA_TranslucentBackground)
        self.drawing_proxy = self.scene.addWidget(self.drawing_label)
        self.drawing_proxy.setZValue(1)
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
            
            # 1. Process bounding box if exists
            if 'bbox' in ann and len(ann['bbox']) >= 4:
                x, y, w, h = ann['bbox']
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
                    kpt_name = (self.project_config['keypoints_name'][kpt_id] 
                            if ('keypoints_name' in self.project_config and 
                                kpt_id < len(self.project_config['keypoints_name']))
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
                self.drawing_label.targets = self.labels_cache[self.current_image]['targets']
                self.drawing_label._commit_changes()
            
            # Set view parameters
            self.FrameDisplay.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.statusBar().showMessage(f"Displaying: {key}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")

    # Additional methods (unchanged from original)
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
                self.project_config = yaml.safe_load(f)
            
            self.EditKeypoints.setEnabled(True)
            self.EditSkeleton.setEnabled(True)
            self.EditClasses.setEnabled(True)
            
            # Validate required sections
            required = ['keypoints_name', 'classes_name', 'skeleton']
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
            self.project_config['keypoints_name'])
        self.ClassesSelection.addItems(
            self.project_config['classes_name'])
        
        skeleton_links = [
            f"{src}->{dest}" for src, dest in self.project_config['skeleton']
        ]
        self.SkeletonSelection.addItems(skeleton_links)
        
    def _ConfigureDisplay(self):
        """Copy all relevant properties between QGraphicsView instances"""
        # Core properties
        self.FrameDisplay.setAlignment(Qt.AlignCenter)
        self.FrameDisplay.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.FrameDisplay.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        # Enable high-quality rendering
        self.FrameDisplay.setRenderHint(QPainter.Antialiasing)
        self.FrameDisplay.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Initialize zoom parameters
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

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

    def onEditParameters(self):
        """Handle parameter editing with state tracking"""
        button = self.sender()
        button_map = {
            self.EditKeypoints: 'keypoints',
            self.EditClasses: 'classes',
            self.EditSkeleton: 'skeleton',
        }

        config_type = button_map.get(button)

        self._edit_mode = not self._edit_mode
        
        if self._edit_mode:
            self._HandleConfigDisplay(config_type)
            button.setText(f"Save {config_type.title()}")
            self.ConfigureDisplay.itemChanged.connect(
                self.onConfigureEdited,
                Qt.QueuedConnection)
        else:
            self.ConfigureDisplay.blockSignals(True)
            try:
                self._SaveConfigChanges(config_type)
                button.setText(f"Edit {config_type.title()}")
                self.ConfigureDisplay.itemChanged.disconnect()
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))
            finally:
                self.ConfigureDisplay.blockSignals(False)

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
                self.project_config['keypoints_name'] = updated_data
            elif config_type =='skeleton':
                self.project_config['skeleton'] = updated_data
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.project_config, 
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
            'keypoints': ('keypoints_name', ['Keypoint ID', 'Keypoint Name']),
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
                key = item.text(0)
                if self.EditClasses.text().startswith('Save'):
                    self.project_config['classes_name'][key] = new_value
                elif self.EditKeypoints.text().startswith('Save'):
                    self.project_config['keypoints_name'][key] = new_value
                elif self.EditSkeleton.text().startswith('Save'):
                    if isinstance(new_value, list) and len(new_value) == 2:
                        self.project_config['skeleton'][key] = new_value
        finally:
            self._edit_lock = False

    def convert_value(self, value):
        """
        Converts a string value to appropriate Python type.
        Now also handles skeleton link format "X->Y".
        """
        # Return non-strings immediately
        if not isinstance(value, str):
            return value
            
        value = value.strip()  # Normalize input (don't lower() for skeleton links)
        
        # Handle empty/None cases
        if not value or value.lower() in ("none", "null"):
            return None
            
        # Handle booleans
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Handle skeleton links (format "X -> Y")
        if '->' in value:
            try:
                parts = [x.strip() for x in value.split('->')]
                return [int(parts[0]), int(parts[1])]
            except (ValueError, IndexError):
                return value  # Return original if conversion fails
        
        # Numeric conversion attempts
        try:
            return int(value)  # Try integer first
        except ValueError:
            try:
                return float(value)  # Then try float
            except ValueError:
                return value  # Return original string if all conversions fail
    

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
        """Save current annotations to file"""
        if not self.drawing_label.targets:
            QMessageBox.warning(self, "Warning", "No annotations to save")
            return
        
        if self.current_image:
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
            annotation = {
                "id": target.id,
                "image_id": self.current_image_index,
                "category_id": target.bounding_rect.get('category_id'),
                "bbox": target.bounding_rect.get('bbox'),
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
        with open(label_file, 'w') as f:
            json.dump(coco_dict, f, indent=4)
    
    def onDeleteLabel(self):
        """Delete current annotations"""
        if not self.drawing_label.targets:
            QMessageBox.warning(self, "Warning", "No annotations to delete")
            return
        self.drawing_label.targets = []
        self.drawing_label._commit_changes()

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
        if hasattr(self, 'drawing_label'):
            self.SkeletonSelection.addItem(f"{value[0]}->{value[1]}")
            if value not in self.project_config['skeleton']:
                self.project_config['skeleton'].append(value)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.project_config, 
                        f, 
                        indent=2, 
                        sort_keys=False, 
                        default_flow_style=False
                    )
    
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
        """Handle closing of the save dialog"""
        self.scene.clear()
        self.drawing_label = None
        event.accept()

    def resizeEvent(self, event):
        """Maintain aspect ratio when resizing window"""
        super().resizeEvent(event)
        if self.scene.items():
            self.FrameDisplay.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

def main():
    app = QApplication(sys.argv)
    window = AnimalPoseAnnotatorPage()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()