from PySide6.QtCore import Qt, QFile, QTextStream
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow,
                               QMessageBox, QTreeWidgetItem, QTreeWidgetItemIterator,
                               QGraphicsScene, QGraphicsView)
from PySide6.QtGui import QPixmap, QPainter, QStandardItemModel
from animalposetracker.gui import (DARK_THEME_PATH, LIGHT_THEME_PATH,
                                  LOGO_PATH_TRANSPARENT, LOGO_PATH,
                                  LOGO_SMALL_PATH, WELCOME_PATH)
import os
import sys
from pathlib import Path
import yaml
from animalposetracker.data import AnimalPoseAnnotation, AnimalPoseAnnotator

from .ui_animalposeannotator import Ui_AnimalPoseAnnotator
from .utils import ZoomableGraphicsView, DrawingBoard

class AnimalPoseAnnotatorPage(QMainWindow, Ui_AnimalPoseAnnotator):
    """Main application window for animal pose annotation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setupConnections()
        self.initialize_controls()
        self.current_drawing_mode = "none"
        self.annotation_targets = []

    def initialize_controls(self):
        """Initialize UI controls and state"""
        os.chdir(Path.cwd())
        self.project_config = {}
        self.images = {}
        self.current_image_index = -1
        self.current_image_path = ""
        self.sorted_keys = []
        self._CreateGraphicsScene()
        self.ConfigureDisplay.header().close()
        self._edit_mode = False

    def setupConnections(self):
        """Connect all UI signals to their corresponding slot functions"""
        # File operations
        self.actionOpenConfigureFile.triggered.connect(self.onOpenConfigFile)
        self.actionOpenFileFolder.triggered.connect(self.onOpenFileFolder)
        self.actionSaveLabel.triggered.connect(self.onSaveLabel)
        
        # Navigation controls
        self.actionNextFrame.triggered.connect(self.onNextFrame)
        self.actionPreviousFrame.triggered.connect(self.onPreviousFrame)
        
        # Annotation tools
        self.actionDrawBBox.triggered.connect(lambda: self.setDrawingMode("rect"))
        self.actionDrawPoint.triggered.connect(lambda: self.setDrawingMode("point"))
        self.actionDrawLine.triggered.connect(lambda: self.setDrawingMode("line"))
        self.actionBuildSkeleton.triggered.connect(lambda: self.setDrawingMode("bline"))
        self.actionNoLabel.triggered.connect(lambda: self.setDrawingMode("none"))
        self.actionDrawLine.setEnabled(False)
        self.actionDrawPoint.setEnabled(False)
        
        # Editing operations
        self.actionAdd.triggered.connect(self.onAddItem)
        self.actionDelete.triggered.connect(self.onDeleteItem)
        self.actionDeleteLabel.triggered.connect(self.onDeleteLabel)
        self.actionUndoLasted.triggered.connect(self.onUndoLastAction)
        
        # Configuration editing
        self.EditClasses.clicked.connect(self.onEditParameters)
        self.EditSkeleton.clicked.connect(self.onEditParameters)
        self.EditKeypoints.clicked.connect(self.onEditParameters)
        
        # View settings
        self.actionDark.triggered.connect(lambda: self.onChangeTheme("dark"))
        self.actionLight.triggered.connect(lambda: self.onChangeTheme("light"))
        
        # Help
        self.actionHelp.triggered.connect(self.onShowHelp)
        
        # Combo box changes
        self.KeypointsSeletion.currentIndexChanged.connect(self.onKeypointSelectionChanged)
        self.ClassesSelection.currentIndexChanged.connect(self.onClassSelectionChanged)
        self.SkeletonSelection.currentIndexChanged.connect(self.onSkeletonSelectionChanged)
        self.Visible.currentIndexChanged.connect(self.onVisibleChanged)
        self.Visible.addItems(['Invisible', 'Occluded', 'Visible'])
        self.Visible.setCurrentIndex(2)
        # Tree view selections
        self.ConfigureDisplay.itemSelectionChanged.connect(self.onConfigureDisplaySelectionChanged)

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
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', 
                                '.gif', '.tif', '.tiff', '.webp'}
        folder = Path(folder_path)
        
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in supported_extensions:
                self.images[img_file.name] = str(img_file)
        
        if self.images:
            self.statusBar().showMessage(f"Loaded {len(self.images)} images")
            self.sorted_keys = sorted(self.images.keys())
            self.current_image_index = 0
            self._DisplayCurrentImage()
        else:
            QMessageBox.warning(self, "Warning", "No supported images found")

    def _DisplayCurrentImage(self):
        """Display current image with annotations"""
        if not self.images:
            self.scene.clear()
            return

        key = self.sorted_keys[self.current_image_index]
        image_path = Path(self.images[key])
        
        try:
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
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use safe loader instead of FullLoader
                self.project_config = yaml.safe_load(f)
            
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
        self.current_image_index = max(0, min(self.current_image_index, len(self.images)-1))
        self._DisplayCurrentImage()

    def onNextFrame(self):
        """Display the next image (with wrap-around)"""
        if not self._check_images_loaded():
            return
        self._advance_image_index(1)

    def onPreviousFrame(self):
        """Display the previous image (with wrap-around)"""
        if not self._check_images_loaded():
            return
        self._advance_image_index(-1)

    def _check_images_loaded(self):
        """Helper method to check if images are loaded"""
        if not self.images:
            self.statusBar().showMessage("No images loaded")
            return False
        return True

    def onAddItem(self):
        """
        Handle adding new annotation item.
        
        Creates a new annotation element in the current frame.
        """
        print("Adding item")
        # Implementation for adding items goes here

    def onDeleteItem(self):
        """
        Handle deleting selected item.
        
        Removes the currently selected annotation element.
        """
        print("Deleting item")
        # Implementation for deleting items goes here

    def onDeleteLabel(self):
        """
        Handle deleting current label.
        
        Clears all annotations for the current frame.
        """
        print("Deleting label")
        # Implementation for deleting labels goes here

    def onUndoLastAction(self):
        """
        Handle undoing last action.
        
        Reverts the most recent annotation change.
        """
        print("Undoing last action")
        # Implementation for undo functionality goes here

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
                self.onConfigureEdited)
        else:
            try:
                self._SaveConfigChanges(config_type)
                button.setText(f"Edit {config_type.title()}")
                self.ConfigureDisplay.itemChanged.disconnect()
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    def _SaveConfigChanges(self, config_type):
        """Validate and save configuration changes"""
        updated_data = []
        
        iterator = QTreeWidgetItemIterator(self.ConfigureDisplay)
        while iterator.value():
            item = iterator.value()
            if config_type == 'skeleton':
                try:
                    src, dest = item.text(1).split('->')
                    updated_data.append((int(src.strip()), int(dest.strip())))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid skeleton format: {item.text(1)}") from e
            else:
                updated_data.append(item.text(1))
            
            iterator +=1
        
        # Update config and refresh UI
        self.project_config[config_type+'name'] = updated_data
        self._populate_comboboxes()
    
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
                connection_text = f"{item[0]}->{item[1]}"
                list_item.setText(1, connection_text)
            else:
                list_item.setText(1, str(item))
            
            list_item.setFlags(list_item.flags() | Qt.ItemIsEditable)
            list_item.setData(0, Qt.UserRole, idx)


    def onConfigureEdited(self, item, column):
        """Slot function for handling other parameters editing"""
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
            if theme == "dark":
                self.Logo.setPixmap(QPixmap(LOGO_PATH_TRANSPARENT))
            elif theme == "light":
                # self.Logo.setPixmap(QPixmap(LOGO_PATH))
                self.Logo.setPixmap(QPixmap(LOGO_PATH_TRANSPARENT))
        except Exception as e:
            QMessageBox.warning(self, "Theme Error", str(e))

    def onShowHelp(self):
        """
        Handle showing help information.
        
        Displays application help documentation.
        """
        print("Showing help")

    def onKeypointSelectionChanged(self, index):
        """
        Handle keypoint selection change.
        
        Args:
            index (int): New selected index in keypoints combo box
        """
        print(f"Keypoint selection changed to index: {index}")
        # Implementation for keypoint selection change goes here

    def onClassSelectionChanged(self):
        """Handle class selection changes"""
        if self.drawing_label.current_target:
            self.drawing_label.current_target.class_id = self.ClassesSelection.currentIndex()
            self.drawing_label.current_target.class_name = self.ClassesSelection.currentText()
            self.drawing_label._commit_changes()

    def onSkeletonSelectionChanged(self, index):
        """
        Handle skeleton selection change.
        
        Args:
            index (int): New selected index in skeleton combo box
        """
        print(f"Skeleton selection changed to index: {index}")
        # Implementation for skeleton selection change goes here
    
    def onVisibleChanged(self, index):
        """Handle visibility change for current target"""
        pass

    def onConfigureDisplaySelectionChanged(self):
        """
        Handle configure display selection change.
        
        Updates UI when items in configuration tree are selected.
        """
        selected_items = self.ConfigureDisplay.selectedItems()
        if selected_items:
            print(f"Configure display selection changed: {selected_items[0].text(0)}")
            # Implementation for config display selection goes here

    def onLabelInfoSelectionChanged(self):
        """
        Handle label information selection change.
        
        Updates UI when items in label info view are selected.
        """
        selected_indexes = self.LabelInformationView.selectedIndexes()
        if selected_indexes:
            print(f"Label info selection changed: {selected_indexes[0].data()}")
            # Implementation for label info selection goes here
    
    def onSaveLabel(self):
        """Save current annotations to file"""
        if not self.drawing_label.targets:
            QMessageBox.warning(self, "Warning", "No annotations to save")
            return
    
        image = {
                "id": self.current_image_index,
                "file_name": self.sorted_keys[self.current_image_index],
                "width": self.drawing_label.main_pixmap.width(),
                "height": self.drawing_label.main_pixmap.height(),
                "license": "",
                "date_captured": "",
        }
        annotations = []
        for target in self.drawing_label.targets:
            if target.keypoints:
                keypoints = [target.keypoints[kp_id]["pos"] for kp_id in sorted(target.keypoints.keys())]
            else:
                keypoints = []
            annotation = {
                "id": target.id,
                "image_id": self.current_image_index,
                "category_id": target.bounding_rect.get('category_id'),
                "bbox": target.bounding_rect.get('bbox'),
                "area": target.bounding_rect.get('area'),
                "iscrowd": int(self.IsCrowd.isChecked()),
                "keypoints": keypoints, 
            }
            annotations.append(annotation)
        coco_dict = {
            "images": [image],
            "annotations": annotations,
        }
        print(coco_dict)

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