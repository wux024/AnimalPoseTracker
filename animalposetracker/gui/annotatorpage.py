from PySide6.QtCore import  Qt, QPoint, QRect
from PySide6.QtWidgets import  (QMenu, QApplication, QFileDialog, QMainWindow,
                                 QMessageBox, QTreeWidget, QTreeWidgetItem, 
                                 QTreeWidgetItemIterator, QLabel, QGraphicsScene, 
                                 QGraphicsView
                                 )
from PySide6.QtGui import QCursor, QPixmap, QPen, QColor, QPainter, QWheelEvent
import os
import sys
from pathlib import Path
from collections import deque
import yaml
from animalposetracker.data import AnimalPoseAnnotation, AnimalPoseAnnotator

from .ui_animalposeannotator import Ui_AnimalPoseAnnotator

class DrawingLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.pixmap = QPixmap(800, 600)  # Create blank canvas
        self.pixmap.fill(Qt.white)       # Fill with white background
        self.setPixmap(self.pixmap)
        
        # Store drawing elements
        self.lines = []      # List of lines: [(start_pos, end_pos), ...]
        self.points = []     # List of points: [QPoint, ...]
        self.rects = []      # List of rectangles: [QRect, ...]
        
        # Current drawing mode (default: point)
        self.drawing_mode = "point"  # "point" | "line" | "rect"
        
        # Temporary storage during drawing
        self.start_pos = None  # Stores initial click position

    def setDrawingMode(self, mode):
        """Set drawing mode: 'point', 'line', or 'rect'"""
        self.drawing_mode = mode

    def mousePressEvent(self, event):
        """Handle mouse press: record starting position"""
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            
            # If in point mode, add point immediately
            if self.drawing_mode == "point":
                self.points.append(event.pos())
                self.update()  # Trigger repaint

    def mouseMoveEvent(self, event):
        """Handle mouse movement: show temporary preview"""
        if self.start_pos and event.buttons() & Qt.LeftButton:
            # For line mode: update temporary end position
            if self.drawing_mode == "line":
                self.temp_end_pos = event.pos()
                self.update()
            
            # For rect mode: update temporary rectangle
            elif self.drawing_mode == "rect":
                self.temp_rect = QRect(self.start_pos, event.pos()).normalized()
                self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release: finalize the shape"""
        if event.button() == Qt.LeftButton and self.start_pos:
            end_pos = event.pos()
            
            # Store completed line
            if self.drawing_mode == "line":
                self.lines.append((self.start_pos, end_pos))
            
            # Store completed rectangle
            elif self.drawing_mode == "rect":
                self.rects.append(QRect(self.start_pos, end_pos).normalized())
            
            self.start_pos = None  # Reset starting position
            self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Render all drawing elements"""
        super().paintEvent(event)
        
        # Create painter targeting the pixmap
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable anti-aliasing
        
        # Draw all points (red, 5px)
        pen = QPen(QColor(255, 0, 0), 5)
        painter.setPen(pen)
        for point in self.points:
            painter.drawPoint(point)
        
        # Draw all lines (blue, 2px)
        pen = QPen(QColor(0, 0, 255), 2)
        painter.setPen(pen)
        for line in self.lines:
            painter.drawLine(*line)
        
        # Draw all rectangles (green, 2px)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for rect in self.rects:
            painter.drawRect(rect)
        
        # Draw temporary preview during mouse drag
        if hasattr(self, 'start_pos') and self.start_pos:
            if self.drawing_mode == "line" and hasattr(self, 'temp_end_pos'):
                pen = QPen(QColor(0, 0, 255), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(self.start_pos, self.temp_end_pos)
            
            elif self.drawing_mode == "rect" and hasattr(self, 'temp_rect'):
                pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(self.temp_rect)
        
        painter.end()
        self.setPixmap(self.pixmap)  # Update display

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        """
        Custom QGraphicsView with mouse wheel zoom functionality.
        
        Features:
        - Smooth zoom centered at mouse position
        - Configurable zoom limits and step size
        - High-quality rendering with antialiasing
        """
        super().__init__(parent)
        
        # Zoom configuration
        self.zoom_factor = 1.0      # Current zoom level (1.0 = 100%)
        self.zoom_step = 0.1        # Zoom increment per wheel step
        self.min_zoom = 0.1         # Minimum zoom level (10%)
        self.max_zoom = 10.0        # Maximum zoom level (1000%)
        
        # View optimization settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)  # Zoom at mouse position
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Optional: Enable drag-to-pan behavior
        # self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event: QWheelEvent):
        """
        Handle mouse wheel events for zooming.
        
        Args:
            event: QWheelEvent containing wheel movement data
        """
        # Ignore if no scene or items are loaded
        if not self.scene() or not self.scene().items():
            return
            
        # Calculate zoom direction (positive = zoom in, negative = zoom out)
        zoom_direction = 1 if event.angleDelta().y() > 0 else -1
        
        # Store mouse scene position before zoom
        old_pos = self.mapToScene(event.position().toPoint())
        
        # Calculate new zoom factor with constraints
        new_zoom = self.zoom_factor * (1 + self.zoom_step * zoom_direction)
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        # Skip if zoom level didn't change
        if new_zoom == self.zoom_factor:
            return
            
        # Apply zoom transformation
        zoom_ratio = new_zoom / self.zoom_factor
        self.scale(zoom_ratio, zoom_ratio)
        self.zoom_factor = new_zoom
        
        # Adjust view to maintain mouse position
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        
        # Accept the event to prevent propagation
        event.accept()

class AnimalPoseAnnotatorPage(QMainWindow, Ui_AnimalPoseAnnotator):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.initialize_controls()
        self.setupConnections()
    

    def initialize_controls(self):
        self.project_config = {}
        self.images = {}
        self.current_image_index = -1
        self.sorted_keys = []
        self._CreateGraphicsScene()

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
        self.actionDrawBBox.triggered.connect(self.onDrawBBox)
        self.actionDrawPoint.triggered.connect(self.onDrawPoint)
        self.actionDrawLine.triggered.connect(self.onDrawLine)
        self.actionNoLabel.triggered.connect(self.onNoLabel)
        
        # Editing operations
        self.actionAdd.triggered.connect(self.onAddItem)
        self.actionDelete.triggered.connect(self.onDeleteItem)
        self.actionDeleteLabel.triggered.connect(self.onDeleteLabel)
        self.actionUndoLasted.triggered.connect(self.onUndoLastAction)
        
        # Configuration editing
        self.EditClasses.clicked.connect(self.onEditParameters)
        
        # View settings
        self.actionDark.triggered.connect(self.onSetDarkTheme)
        self.actionLight.triggered.connect(self.onSetLightTheme)
        
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
        self.FrameDisplay.setScene(self.scene)

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
        drawing_proxy = self.scene.addWidget(self.drawing_label)
        drawing_proxy.setZValue(1)  # Make sure drawings appear above the image

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
        """
        Handle opening a configuration file.
        
        Opens a file dialog to select configuration file and loads it.
        Supported formats: JSON, YAML
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Config File", "", "Config Files (*.yaml *.yml)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.project_config = yaml.load(f, Loader=yaml.FullLoader)

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
        self._display_current_image()

    def _display_current_image(self):
        """Display the current image with bounds checking."""
        if not self.images:
            return
        
        # Handle index bounds
        if self.current_image_index < 0:
            self.current_image_index = 0
        elif self.current_image_index >= len(self.images):
            self.current_image_index = len(self.images) - 1
        
        key = self.sorted_keys[self.current_image_index]
        image_path = self.images[key]
        
        self.scene.clear()
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            self.statusBar().showMessage(f"Failed to load image: {image_path}")
            return

        view_height = self.FrameDisplay.viewport().height()
        img_height = pixmap.height()

        pixmap_item = self.scene.addPixmap(pixmap)
        scene_rect = pixmap_item.boundingRect()
        self.scene.setSceneRect(scene_rect)

        if img_height < view_height:
            top_margin = (view_height - img_height) / 2
            self.scene.setSceneRect(0, -top_margin, 
                                  scene_rect.width(), 
                                  view_height)
        
        self.FrameDisplay.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.statusBar().showMessage(f"Showing image {key} ({image_path})")
    
    def onSaveLabel(self):
        """
        Handle saving current label data.
        
        Saves the current annotation state to file.
        """
        print("Saving current label")
        # Implementation for saving label goes here

    def onNextFrame(self):
        """Display the next image (with wrap-around)"""
        if not self._check_images_loaded():  # 提取重复逻辑
            return
        
        self.current_image_index += 1
        if self.current_image_index >= len(self.images):
            self.current_image_index = 0
        
        self._display_current_image()

    def onPreviousFrame(self):
        """Display the previous image (with wrap-around)"""
        if not self._check_images_loaded():  # 提取重复逻辑
            return
        
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.images) - 1
        
        self._display_current_image()

    def _check_images_loaded(self):
        """Helper method to check if images are loaded"""
        if not self.images:
            self.statusBar().showMessage("No images loaded")
            return False
        return True
    
    def resizeEvent(self, event):
        """Handle window resize to maintain image aspect ratio"""
        super().resizeEvent(event)
        if hasattr(self, 'scene') and self.scene.items():
            self.FrameDisplay.fitInView(self.scene.items()[0], Qt.KeepAspectRatio)

    def onDrawBBox(self):
        """
        Handle setting bounding box drawing mode.
        
        Activates the bounding box annotation tool.
        """
        self.drawing_label.setDrawingMode("rect")

    def onDrawPoint(self):
        """
        Handle setting point drawing mode.
        
        Activates the point annotation tool.
        """
        self.drawing_label.setDrawingMode("point")

    def onDrawLine(self):
        """
        Handle setting line drawing mode.
        
        Activates the line annotation tool.
        """
        self.drawing_label.setDrawingMode("line")

    def onNoLabel(self):
        """
        Handle setting no label mode.
        
        Deactivates all annotation tools.
        """
        print("Setting no label mode")
        # Implementation for drawing mode goes here

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
        """
        Handle editing class definitions.
        
        Opens dialog for modifying available annotation classes.
        """
        button = self.sender()
        BUTTON_TO_WIDGET_MAP = {
        self.EditKeypoints: 'kepoints',
        self.EditClasses: 'classes',
        self.EditSkeleton: 'skeleton',
        }
        selection = BUTTON_TO_WIDGET_MAP[button]
        current_text = button.text()
        is_edit_mode = current_text.startswith('Edit')
        if is_edit_mode:
            if selection == 'classes':
                self._handleEditClass()
            elif selection == 'keypoints':
                self._handleEditKeypoint()
            elif selection =='skeleton':
                self._handleEditSkeleton()
            new_button_text = f"Save {selection.capitalize()}"
        else:
            # save changes and switch to edit mode
            self._handleSaveConfig()
            new_button_text = f"Edit {selection.capitalize()}"

        button.setText(new_button_text)

    def _handleEditConfig(self):
        pass

    def _handleEditClass(self):
        classes_name = self.project_config.get('classes_name', [])
        self.ConfigureDisplay.clear()
        self.ConfigureDisplay.setHeaderLabels(['Classes ID', 'Classes Name'])
        for i, name in enumerate(classes_name):
            item = QTreeWidgetItem(self.ConfigureDisplay)
            item.setText(0, str(i))
            item.setText(1, name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.se

    def _handleEditKeypoint(self):    
        keypoints_name = self.project_config.get('keypoints_name', [])
        self.ConfigureDisplay.clear()
        self.ConfigureDisplay.setHeaderLabels(['Keypoints ID', 'Keypoints Name'])
        for i, name in enumerate(keypoints_name):
            item = QTreeWidgetItem(self.ConfigureDisplay)
            item.setText(0, str(i))
            item.setText(1, name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(0, Qt.UserRole, i)

    def _handleEditSkeleton(self):
        skeleton = self.project_config.get('skeleton', [])
        self.ConfigureDisplay.clear()
        self.ConfigureDisplay.setHeaderLabels(['Link ID', 'Link'])
        for i, link in enumerate(skeleton):
            item = QTreeWidgetItem(self.ConfigureDisplay)
            item.setText(0, str(i))
            item.setText(1, f"{link[0]} -> {link[1]}")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(0, Qt.UserRole, i)


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

    def onSetDarkTheme(self):
        """
        Handle setting dark theme.
        
        Applies dark color scheme to the application.
        """
        print("Setting dark theme")
        # Implementation for theme application goes here

    def onSetLightTheme(self):
        """
        Handle setting light theme.
        
        Applies light color scheme to the application.
        """
        print("Setting light theme")
        # Implementation for theme application goes here

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

    def onClassSelectionChanged(self, index):
        """
        Handle class selection change.
        
        Args:
            index (int): New selected index in classes combo box
        """
        print(f"Class selection changed to index: {index}")
        # Implementation for class selection change goes here

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