from PySide6.QtCore import  Qt, QFile, QTextStream
from PySide6.QtWidgets import  ( QApplication, QFileDialog, QMainWindow,
                                 QMessageBox, QTreeWidgetItem, QTreeWidgetItemIterator, 
                                 QGraphicsScene, QGraphicsView)
from PySide6.QtGui import QPixmap, QPainter
from animalposetracker.gui import (DARK_THEME_PATH, LIGHT_THEME_PATH, 
                                   LOGO_PATH_TRANSPARENT, LOGO_PATH, 
                                   LOGO_SMALL_PATH, WELCOME_PATH)
import os
import sys
from pathlib import Path
import yaml
from animalposetracker.data import AnimalPoseAnnotation, AnimalPoseAnnotator

from .ui_animalposeannotator import Ui_AnimalPoseAnnotator
from .utils import ZoomableGraphicsView, DrawingLabel

class AnimalPoseAnnotatorPage(QMainWindow, Ui_AnimalPoseAnnotator):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.initialize_controls()
        self.setupConnections()
    

    def initialize_controls(self):
        os.chdir(Path.cwd())
        self.project_config = {}
        self.images = {}
        self.current_image_index = -1
        self.sorted_keys = []
        self._CreateGraphicsScene()
        self.ConfigureDisplay.clear()
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
        self.actionDrawBBox.triggered.connect(
            lambda: self.drawing_label.setDrawingMode("rect"))
        self.actionDrawPoint.triggered.connect(
            lambda: self.drawing_label.setDrawingMode("point"))
        self.actionDrawLine.triggered.connect(
            lambda: self.drawing_label.setDrawingMode("line"))
        self.actionNoLabel.triggered.connect(
            lambda: self.drawing_label.setDrawingMode("none"))
        
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
        
        # Tree view selections
        self.ConfigureDisplay.itemSelectionChanged.connect(self.onConfigureDisplaySelectionChanged)


    def _CreateGraphicsScene(self):
        """Create the central graphics view"""

        self._ReplaceGraphicsView()

        self.scene = QGraphicsScene()

        self.FrameDisplay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.FrameDisplay.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.FrameDisplay.setRenderHints(
            QPainter.Antialiasing | 
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing
        )

        self.FrameDisplay.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.FrameDisplay.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        # Add drawing label to the scene
        self.drawing_label = DrawingLabel()
        self.drawing_label.setAttribute(Qt.WA_DeleteOnClose, False)
        self.drawing_proxy = self.scene.addWidget(self.drawing_label)
        self.drawing_proxy.setZValue(1)  # Make sure drawings appear above the image
        self.drawing_proxy.setPos(0, 0)

        self.FrameDisplay.setScene(self.scene)

        self._ConfigureDisplay()
        
    
    def _ReplaceGraphicsView(self):
        """Safely replaces the default QGraphicsView with our custom version"""
        original_view = self.FrameDisplay
        
        # Create custom view with same parent
        self.FrameDisplay = ZoomableGraphicsView(self.FrameDisplayGroup)
        
        # Copy all properties from original view
        self.FrameDisplay.setObjectName(original_view.objectName())
        self.FrameDisplay.setStyleSheet(original_view.styleSheet())
        
        # Replace in layout
        self.FrameDisplayGroupLayout.replaceWidget(original_view, self.FrameDisplay)
        original_view.deleteLater()
        
    
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
            f"{src} -> {dest}" for src, dest in self.project_config['skeleton']
        ]
        self.SkeletonSelection.addItems(skeleton_links)

    def onOpenFileFolder(self):
        """Handle opening a file folder using pathlib."""
        folder_path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if not folder_path:  # User cancelled
            return
        
        self.images.clear()
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp',
                            '.gif', '.tif', '.tiff', '.webp'}
        
        folder = Path(folder_path)
        
        # More efficient single pass through files
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in supported_extensions and img_file.is_file():
                # Use full filename (with extension) as key to avoid collisions
                key = img_file.name
                self.images[key] = str(img_file)
        
        if self.images:
            self.statusBar().showMessage(f"Loaded {len(self.images)} images from {folder_path}")
        else:
            self.statusBar().showMessage("No supported image files found in folder")
        
        self.sorted_keys = sorted(self.images.keys())
        self.current_image_index = 0 if self.images else -1
        self._DisplayCurrentImage()

    def _DisplayCurrentImage(self):
        """Display the current image with proper error handling and drawing layer setup."""
        if not self.images:
            self.scene.clear()
            self.drawing_label = None
            self.drawing_proxy = None
            return

        # Validate and adjust current image index
        self.current_image_index = max(0, min(self.current_image_index, len(self.images)-1))
        key = self.sorted_keys[self.current_image_index]
        image_path = Path(self.images[key])

        self.scene.clear()
        self.drawing_label = None
        self.drawing_proxy = None

        try:
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                raise ValueError("Invalid image")
            pixmap_item = self.scene.addPixmap(pixmap)
            scene_rect = pixmap_item.boundingRect()
            self.scene.setSceneRect(scene_rect)

            self.drawing_label = DrawingLabel()
            self.drawing_label.setAttribute(Qt.WA_TranslucentBackground)
            self.drawing_label.setFixedSize(pixmap.size())
            self.drawing_proxy = self.scene.addWidget(self.drawing_label)
            self.drawing_proxy.setZValue(1)
            self.drawing_proxy.setPos(pixmap_item.pos())

            self.FrameDisplay.fitInView(scene_rect, Qt.KeepAspectRatio)
            self.statusBar().showMessage(f"Displaying: {key}")

        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.current_image_index += 1
            if self.current_image_index >= len(self.images):
                self.current_image_index = 0
            self._DisplayCurrentImage() 

    def _advance_image_index(self, step):
        """Safely advance image index with wrap-around handling."""
        if not self.images:
            return
        self.current_image_index += step
        self.current_image_index = max(0, min(self.current_image_index, len(self.images)-1))
        self._DisplayCurrentImage()

    def resizeEvent(self, event):
        """Maintain aspect ratio when window is resized."""
        super().resizeEvent(event)
        if self.scene and not self.scene.itemsBoundingRect().isEmpty():
            self.FrameDisplay.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def onSaveLabel(self):
        """
        Handle saving current label data.
        
        Saves the current annotation state to file.
        """
        if not self.drawing_label.targets:
            QMessageBox.warning(self, "Warning", "No annotations to save")
            return
        
        annotations = []
        for target in self.drawing_label.targets:
            annotation = {
                "id": 0,
                "image_id": self.sorted_keys[self.current_image_index],
                "category_id": target.class_id,
                "bbox": {
                    "x": target.rect.x(),
                    "y": target.rect.y(),
                    "width": target.rect.width(),
                    "height": target.rect.height()
                },
                "area": target.rect.width() * target.rect.height(),
                "keypoints": [(p.x(), p.y()) for p in target.points],

            }
            annotations.append(annotation)
        print(annotations)

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
            idx = int(item.text(0))
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
                connection_text = f"{item[0]} -> {item[1]}"
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
        """
        Handle class selection change.
        
        Args:
            index (int): New selected index in classes combo box
        """
        if self.drawing_label.current_target:
            self.drawing_label.current_target.class_id = (
                self.ClassesSelection.currentIndex())
            self.drawing_label.current_target.class_name = (
                self.ClassesSelection.currentText())
            self.drawing_label._draw_permanent()
        

    def onSkeletonSelectionChanged(self, index):
        """
        Handle skeleton selection change.
        
        Args:
            index (int): New selected index in skeleton combo box
        """
        print(f"Skeleton selection changed to index: {index}")
        # Implementation for skeleton selection change goes here

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

def main():
    app = QApplication(sys.argv)
    window = AnimalPoseAnnotatorPage()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()