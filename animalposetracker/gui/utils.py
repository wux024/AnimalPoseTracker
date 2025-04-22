from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal, QObject
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, 
                              QListWidgetItem, QGraphicsView, QCheckBox, 
                              QLayout, QTreeView, QMenu, QDialog, QPushButton, 
                              QVBoxLayout, QMessageBox)
from PySide6.QtGui import (QPainter, QPixmap, QPen, QColor,QWheelEvent, QBrush, 
                           QStandardItemModel, QStandardItem, QFontMetrics)
from typing import List
import shutil
import json
from sklearn.model_selection import train_test_split
from animalposetracker.gui import COLORS


class ListManager(QObject):
    list_changed = Signal()
    def __init__(self, scroll_area, initial_widgets):
        super().__init__()
        self.scroll_area = scroll_area
        self.container = initial_widgets['container']
        self.v_layout = initial_widgets['v_layout']
        
        # Store all input rows while preserving your exact layout structure
        self.input_rows = [{
            'number_label': initial_widgets['label'],
            'input_field': initial_widgets['input'],
            'row_layout': initial_widgets['h_layout']
        }]
        
        # Configure first row to match your screenshot exactly
        initial_widgets['label'].setText("1.")
        initial_widgets['label'].setAlignment(Qt.AlignCenter)
        initial_widgets['input'].setStyleSheet("""
            QLineEdit {
                border: none;
                border-bottom: 1px solid #ccc;
                padding: 2px 0;
            }
            QLineEdit:focus {
                border-bottom: 1px solid #1E90FF;
            }
        """)
        
        # Connect signals while maintaining your layout constraints
        initial_widgets['input'].textChanged.connect(self._handle_input_change)

    def ensure_single_empty_row(self):
        """Ensures that there is at most one empty row in the list"""
        while len(self.input_rows) > 1:
            last_row = self.input_rows[-1]
            if last_row['input_field'].text().strip():
                break
            self._remove_last_row()
        
        if not self.input_rows[-1]['input_field'].text().strip():
            return  
            
        self._add_input_row()
        self.input_rows[-1]['input_field'].clear()

    def _handle_input_change(self):
        """Adds new row when last input has content, matching your layout style"""
        current_text = self.input_rows[-1]['input_field'].text().strip()
        if current_text and len(self.input_rows) == self.v_layout.count():
            self._add_input_row()
        self.ensure_single_empty_row()
        self.list_changed.emit()

    def _add_input_row(self):
        """Creates new row while preserving your exact layout parameters"""
        row_number = len(self.input_rows) + 1
        
        # Create new layout with your exact specifications
        row_layout = QHBoxLayout()
        row_layout.setSpacing(10)
        row_layout.setContentsMargins(10, 10, -1, -1)
        row_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        
        # Create label matching your original properties
        number_label = QLabel(self.container)
        number_label.setObjectName(f"KeypointNumber{row_number}")
        number_label.setSizePolicy(self.input_rows[0]['number_label'].sizePolicy())
        number_label.setMinimumSize(QSize(10, 25))
        number_label.setMaximumSize(QSize(16777215, 25))
        number_label.setAlignment(Qt.AlignCenter)
        number_label.setText(f"{row_number}.")
        
        # Create input field matching your original properties
        input_field = QLineEdit(self.container)
        input_field.setSizePolicy(self.input_rows[0]['input_field'].sizePolicy())
        input_field.setMinimumSize(QSize(400, 25))
        input_field.setMaximumSize(QSize(16777215, 25))
        input_field.setStyleSheet(self.input_rows[0]['input_field'].styleSheet())
        input_field.textChanged.connect(self._handle_input_change)
        
        # Add to layout while maintaining your structure
        row_layout.addWidget(number_label)
        row_layout.addWidget(input_field)
        self.v_layout.addLayout(row_layout)
        
        # Store reference
        self.input_rows.append({
            'number_label': number_label,
            'input_field': input_field,
            'row_layout': row_layout
        })

    def get_list_data(self):
        """Returns data in format [(1, "text"), (2, "text"), ...]"""
        return [row['input_field'].text().strip() for row in self.input_rows
                if row['input_field'].text().strip()] 

    def clear_inputs(self):
        """Clears all fields while preserving first row layout"""
        while len(self.input_rows) > 1:
            self._remove_last_row()
        self.input_rows[0]['input_field'].clear()
        self.ensure_single_empty_row()
        self.list_changed.emit()

    def _remove_last_row(self):
        """Removes last row while maintaining layout integrity"""
        if len(self.input_rows) <= 1:
            return
        last_row = self.input_rows.pop()
        self._cleanup_layout(last_row['row_layout'])

    def _cleanup_layout(self, layout):
        """Safely removes layout while preserving your structure"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        layout.deleteLater()

class SourceSignal(QObject):
    source_selected = Signal(bool)

class CheckableListWidgetItem(QListWidgetItem):
    def __init__(self, text):
        super().__init__()
        self.checked_signal = SourceSignal()

        self.widget = QWidget()
        self.checkbox = QCheckBox(text)
        self.checkbox.setChecked(True)

        layout = QHBoxLayout(self.widget)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.addWidget(self.checkbox)
        
        self.checkbox.stateChanged.connect(self._emit_check_state)
    
    def _emit_check_state(self):
        self.checked_signal.source_selected.emit(self.checkbox.isChecked())


class ZoomableGraphicsView(QGraphicsView):
    zoom_changed = Signal(float)
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
        
        self.setUpdatesEnabled(True)  # Enable view updates

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
        self.zoom_changed.emit(self.zoom_factor)

class DatasetSplitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Dataset Split Ratios")
        self.ratios = (0.8, 0.1, 0.1)  # Default ratios
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        # Training set ratio
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Training set ratio:"))
        self.train_edit = QLineEdit("0.8")
        train_layout.addWidget(self.train_edit)
        layout.addLayout(train_layout)
        
        # Validation set ratio
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation set ratio:"))
        self.val_edit = QLineEdit("0.1")
        val_layout.addWidget(self.val_edit)
        layout.addLayout(val_layout)
        
        # Test set ratio
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test set ratio:"))
        self.test_edit = QLineEdit("0.1")
        test_layout.addWidget(self.test_edit)
        layout.addLayout(test_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.validate_and_accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def validate_and_accept(self):
        try:
            train = float(self.train_edit.text())
            val = float(self.val_edit.text())
            test = float(self.test_edit.text())
            
            if abs(train + val + test - 1.0) > 0.001:
                QMessageBox.warning(self, "Invalid Ratios", "Ratios must sum to 1.0")
                return
                
            if train <= 0 or val <= 0 or test <= 0:
                QMessageBox.warning(self, "Invalid Ratios", "All ratios must be positive")
                return
                
            self.ratios = (train, val, test)
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers")

class DatasetSplitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Dataset Split Ratios")
        self.ratios = (0.8, 0.1, 0.1)  # Default ratios
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Training set ratio
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Training set ratio:"))
        self.train_edit = QLineEdit("0.8")
        train_layout.addWidget(self.train_edit)
        layout.addLayout(train_layout)
        
        # Validation set ratio
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation set ratio:"))
        self.val_edit = QLineEdit("0.1")
        val_layout.addWidget(self.val_edit)
        layout.addLayout(val_layout)
        
        # Test set ratio
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test set ratio:"))
        self.test_edit = QLineEdit("0.1")
        test_layout.addWidget(self.test_edit)
        layout.addLayout(test_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.validate_and_accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def validate_and_accept(self):
        try:
            train = float(self.train_edit.text())
            val = float(self.val_edit.text())
            test = float(self.test_edit.text())
            
            if abs(train + val + test - 1.0) > 0.001:
                QMessageBox.warning(self, "Invalid Ratios", "Ratios must sum to 1.0")
                return
                
            if train <= 0 or val <= 0 or test <= 0:
                QMessageBox.warning(self, "Invalid Ratios", "All ratios must be positive")
                return
                
            self.ratios = (train, val, test)
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers")


def split_dataset_sklearn(image_paths, 
                          label_paths, 
                          ratios=(0.8, 0.1, 0.1), 
                          random_state=42):
    """
    Split dataset using sklearn's train_test_split with proper stratification
        
    Args:
        image_paths: List of Path objects to images
        label_paths: List of Path objects to corresponding labels
        ratios: (train, val, test) ratios (should sum to 1)
        random_state: Random seed for reproducibility
            
    Returns:
        (train_images, train_labels), 
        (val_images, val_labels), 
        (test_images, test_labels)
    """
    # First split into train+val and test
    train_val_ratio = ratios[0] + ratios[1]
    test_ratio = ratios[2]
        
    # First split to separate test set
    train_val_imgs, test_imgs, train_val_lbls, test_lbls = train_test_split(
            image_paths, label_paths, 
            test_size=test_ratio,
            random_state=random_state
    )
        
    # Then split train_val into train and val
    val_ratio = ratios[1] / train_val_ratio  # Adjusted ratio for second split
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_val_imgs, train_val_lbls,
        test_size=val_ratio,
        random_state=random_state
    )
        
    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)

def create_dataset_structure(base_path):
    """Create the required directory structure"""
    dirs = {
        'annotations': base_path / "annotations",
        'images_train': base_path / "images" / "train",
        'images_val': base_path / "images" / "val",
        'images_test': base_path / "images" / "test",
        'labels_train': base_path / "labels" / "train",
        'labels_val': base_path / "labels" / "val",
        'labels_test': base_path / "labels" / "test",
    }
        
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def create_coco_annotations(annotations_input_dir, 
                            annotations_ouput_dir, 
                            train_pairs, val_pairs, test_pairs):
    """
    Generate COCO format annotation files for train/val/test splits while preserving original IDs
    
    Args:
        annotations_dir: Directory to store annotation JSON files
        train_pairs: (train_image_paths, train_label_paths)
        val_pairs: (val_image_paths, val_label_paths)
        test_pairs: (test_image_paths, test_label_paths)
    """
    # Load original COCO annotations
    with open(annotations_input_dir, "r") as f:
        source_ann = json.load(f)
    
    # Create lookup dictionaries:
    # filename → image_id
    # image_id → image_info
    # image_id → [annotations]
    file_name_to_image_id = {img['file_name']: img['id'] for img in source_ann['images']}
    image_id_to_info = {img['id']: img for img in source_ann['images']}
    image_id_to_anns = {img_id: [] for img_id in image_id_to_info}
    
    # Build annotation index (preserve original IDs)
    for ann in source_ann['annotations']:
        image_id_to_anns[ann['image_id']].append(ann)

    # Process each split
    for split_name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        split_ann = {
            "info": source_ann['info'],
            "licenses": source_ann['licenses'],
            "categories": source_ann['categories'],
            "images": [],
            "annotations": []
        }
        
        # Process each image in current split
        for img_path in pairs[0]:  # pairs[0] = image paths
            filename = img_path.name
            
            if filename not in file_name_to_image_id:
                continue  # Skip if not in original annotations
                
            image_id = file_name_to_image_id[filename]
            
            # Add image info (preserve original data)
            split_ann['images'].append(image_id_to_info[image_id])
            
            # Add all annotations (preserve original IDs)
            split_ann['annotations'].extend(image_id_to_anns[image_id])
        
        # Save split annotations
        with open(annotations_ouput_dir / f"{split_name}.json", 'w') as f:
            json.dump(split_ann, f, indent=4)

def copy_split_files(pairs, img_dir, lbl_dir):
    """Copy image-label pairs to their destination directories"""
    for img, lbl in pairs:
        shutil.copy(img, img_dir)
        shutil.copy(lbl, lbl_dir)
class AnnotationTarget:
    """Represents a single annotated object with visual elements"""
    def __init__(self, target_id, color):
        self.id = target_id
        # Display color
        self.color = color
        # List of bounding rectangles
        # {"category_id": int,  "bbox": [x, y, w, h], "area": float, "category_name": str}
        self.bounding_rect = {}   
        # List of keypoints {"keypoint_id": {"keypoint_name": str, "pos": [x, y]}}  
        self.keypoints =  {}   
        # List of line connections {"id": {"skeleton":[id_n,id_m], "pos": [x1, y1, x2, y2]}}    
        self.skeletons = {} 
        self.current_keypoint_id = 0       

class AnnotationViewer(QObject):
    target_deleted = Signal(int)
    keypoints_cleared = Signal(int)
    keypoint_deleted = Signal(int, int)
    skeletons_cleared = Signal(int)
    skeleton_deleted = Signal(int, int)
    def __init__(self, tree_view: QTreeView):
        """Initialize the annotation viewer with a QTreeView
        
        Args:
            tree_view: The QTreeView widget to display annotation hierarchy
        """
        super().__init__()
        self.tree_view = tree_view
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.current_highlight = None  # Stores currently highlighted item
        self._setup_view()

    def _setup_view(self):
        """Configure tree view display properties"""
        self.tree_view.setEditTriggers(QTreeView.NoEditTriggers)
        self.tree_view.setIndentation(20)
        self.tree_view.setColumnWidth(0, 200)  # Fixed width for property column

        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._show_context_menu)
    
    def _show_context_menu(self, pos):
        """Show context menu for tree view items"""
        menu = QMenu()
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self._handle_delete_action)
        menu.exec(self.tree_view.viewport().mapToGlobal(pos))
    
    def _handle_delete_action(self):
        """Handle delete action from context menu"""
        index = self.tree_view.currentIndex()
        if not index.isValid():
            return
        item = self.model.itemFromIndex(index)

        if "Target" in item.text():
            target_id = item.data(Qt.UserRole)
            self._delete_target(target_id)
        elif "Keypoint" in item.text():
            self._delete_keypoint_group(item)
        elif item.parent() and "Keypoint" in item.parent().text():
            self._delete_single_keypoint(item)
        elif "Skeleton" in item.text():
            self._delete_skeleton_group(item)
        elif item.parent() and "Skeleton" in item.parent().text():
            self._delete_single_skeleton(item)
    
    def _delete_target(self, target_id):
        """Delete a target and all associated data"""
        self.target_deleted.emit(target_id)
    
    def _delete_keypoint_group(self, parent_item):
        """Delete a group of keypoints and all associated data"""
        target_id = parent_item.parent().data(Qt.UserRole)
        self.keypoints_cleared.emit(target_id)
    
    def _delete_single_keypoint(self, item):
        """Delete a single keypoint and all associated data"""
        target_id = item.parent().parent().data(Qt.UserRole)
        kpt_id = int(item.text()) 
        self.keypoint_deleted.emit(target_id, kpt_id)
    
    def _delete_skeleton_group(self, parent_item):
        """delete a group of skeletons and all associated data"""
        target_id = parent_item.parent().data(Qt.UserRole)
        self.skeletons_cleared.emit(target_id) 

    def _delete_single_skeleton(self, item):
        """Delete a single skeleton and all associated data"""
        target_id = item.parent().parent().data(Qt.UserRole)
        skeleton_id = int(item.text().split("-")[1])
        self.skeleton_deleted.emit(target_id, skeleton_id)
    
    def display_target(self, targets: List[AnnotationTarget]):
        """Display target data with proper two-column expansion"""

        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Property", "Value"])

        for target in targets:
            # Create root item
            root = QStandardItem(f"Target {target.id}")
            root.setData(target.id, Qt.UserRole)
            self.model.appendRow([root, QStandardItem()])  # Empty value column for root

            # 1. Bounding Box Section
            if target.bounding_rect:
                bbox_header = QStandardItem("Bounding Box")
                root.appendRow([bbox_header, QStandardItem()])  # Empty value
                
                for key, value in target.bounding_rect.items():
                    prop = QStandardItem(key.replace('_', ' ').title())
                    val = QStandardItem(str(value))
                    bbox_header.appendRow([prop, val])  # Proper value in second column

            # 2. Keypoints Section
            if target.keypoints:
                kp_header = QStandardItem(f"Keypoints ({len(target.keypoints)})")
                root.appendRow([kp_header, QStandardItem()])  # Empty value
                
                for kpt_id, kpt_data in target.keypoints.items():
                    kp_parent = QStandardItem(f"{kpt_id}")
                    kp_header.appendRow([kp_parent, QStandardItem()])  # Empty value
                    
                    # Name property-value pair
                    name_prop = QStandardItem("Name")
                    name_val = QStandardItem(kpt_data.get('name', 'Unnamed'))
                    kp_parent.appendRow([name_prop, name_val])
                    
                    # Position property-value pair
                    pos_prop = QStandardItem("Position")
                    x, y, _ = kpt_data['pos']
                    pos_val = QStandardItem(f"({x}, {y})")
                    kp_parent.appendRow([pos_prop, pos_val])

            # 3. Skeletons Section
            if target.skeletons:
                skel_header = QStandardItem(f"Skeletons ({len(target.skeletons)})")
                root.appendRow([skel_header, QStandardItem()])  # Empty value
                
                for skel_id, skel_data in target.skeletons.items():
                    conn_prop = QStandardItem(f"Conn-{skel_id}")
                    src, dst = skel_data['skeleton']
                    conn_val = QStandardItem(f"{src}->{dst}")
                    skel_header.appendRow([conn_prop, conn_val])

            self.tree_view.expandAll()

    
    def highlight_target(self, target_id: int):
        """Highlight a specific target in the tree view
        
        Args:
            target_id: ID of the target to highlight
        """
        # Clear previous highlight
        if self.current_highlight:
            for col in range(self.model.columnCount()):
                self.current_highlight.setBackground(QBrush())
        
        # Find and highlight new target
        items = self.model.findItems(f"Target {target_id}")
        if items:
            self.current_highlight = items[0]
            highlight_color = QBrush(QColor(173, 216, 230))  # Light blue
            for col in range(self.model.columnCount()):
                self.current_highlight.setBackground(highlight_color)
            self.tree_view.scrollTo(self.model.indexFromItem(items[0]))
    
    def clear_highlight(self):
        """Clear all highlighting from the tree view"""
        if self.current_highlight:
            for col in range(self.model.columnCount()):
                self.current_highlight.setBackground(QBrush())
            self.current_highlight = None
    
class DrawingBoard(QLabel):
    """Main canvas for creating and managing annotations"""
    addskeletons = Signal(list)
    def __init__(self, parent=None, 
                 keypoints=None, 
                 classes=None, 
                 skeletons=None,
                 visible=None,
                 labels=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        
        # Canvas state
        self.main_pixmap = QPixmap()
        self.temp_pixmap = QPixmap()
        self.targets = []
        self.reset_canvases()
        
        # Drawing state
        self.drawing_mode = "none"
        self.current_target = None
        self.current_target_id = 0
        self.start_position = None
        self.current_pos = None

        # keypoints, classes and skeleton selection
        self.keypoints = keypoints
        self.classes = classes
        self.skeletons = skeletons
        self.visible = visible
        self.labels = labels
        self.labels.clicked.connect(self._on_label_click)
        self.visible.currentIndexChanged.connect(self._on_visible_change)

        self.labels_view = AnnotationViewer(labels)
        self.labels_view.target_deleted.connect(self._on_target_deleted)
        self.labels_view.keypoints_cleared.connect(self._on_keypoints_cleared)
        self.labels_view.keypoint_deleted.connect(self._on_keypoint_deleted)
        self.labels_view.skeletons_cleared.connect(self._on_skeletons_cleared)
        self.labels_view.skeleton_deleted.connect(self._on_skeleton_deleted)

        # Visual settings
        self.pen_settings = {
            "width": 2,
            "style": Qt.SolidLine,
            "preview_alpha": 180,
            "radius": 3
        }
        self.preview_style = {
            "rect": {
                "color": QColor(255, 0, 0, 120),
                "width": 2,
                "style": Qt.DashLine
            },
            "line": {
                "color": QColor(0, 255, 0, 120),
                "width": 2,
                "style": Qt.DashLine
            }
        }

    def _on_label_click(self, index):
        """Handle label selection from parent widget"""
        target_id = index.data(Qt.UserRole)
        if target_id is not None:
            self.set_current_target(target_id)

    def set_current_target(self, target_id = None):
        """Set current target for drawing operations"""
        if not self.targets:
            self.current_target_id = None
            if hasattr(self, 'labels_view'):
                self.labels_view.clear_highlight()
                return
        
        if target_id is None:
            target_id = self.targets[0].id

        if not any(t.id == target_id for t in self.targets):
            raise ValueError(f"Target {target_id} not found")
            
        self.current_target_id = target_id
        self.current_target = self.targets[self.current_target_id]
        self.keypoints.setCurrentIndex(self.current_target.current_keypoint_id)
        self.labels_view.highlight_target(target_id)

    def reset_canvases(self):
        """Initialize or clear both canvases"""
        size = self.size()
        if size.isValid() and not size.isNull():
            self.main_pixmap = QPixmap(size)
            self.main_pixmap.fill(Qt.transparent)
            self.temp_pixmap = QPixmap(size)
            self.temp_pixmap.fill(Qt.transparent)
            self.update()

    def reset_temp_canvas(self):
        """Clear temporary canvas only"""
        self.temp_pixmap.fill(Qt.transparent)
        self.update()

    def _generate_color_for_id(self, target_id):
        color_index = target_id % len(COLORS) 
        return COLORS[color_index]

    def create_target(self):
        """Create new annotation target with initial properties"""
        new_target = AnnotationTarget(len(self.targets), COLORS[len(self.targets)% len(COLORS)])
        self.current_target_id = new_target.id
        self.targets.append(new_target)
        self.set_current_target(self.current_target_id)
        return new_target

    def mousePressEvent(self, event):
        if self.drawing_mode == "none":
            return
            
        pos = event.pos()
        
        if self.drawing_mode == "rect":
            self._handle_rect_mode(pos)
        elif self.drawing_mode == "point":
            self._handle_point_mode(pos)
        elif self.drawing_mode == "bline":
            self._handle_bline_mode(pos)
                
        super().mousePressEvent(event)
    
    def _handle_rect_mode(self, pos):
        """Process rectangle drawing operations"""
        if self.start_position is None:
            self.start_position = pos
            self.current_target = self.create_target()
            self.current_target._is_temp = True
        else:
            end_pos = pos
            if (abs(self.start_position.x() - end_pos.x()) > 5 or 
                abs(self.start_position.y() - end_pos.y()) > 5):
                rect = QRect(self.start_position, end_pos).normalized()
                self.current_target.bounding_rect = self._get_class_info(rect)
                self.current_target._is_temp = False
                self._commit_changes()

            self.start_position = None
            self.current_target = None
            self.reset_temp_canvas()

    def _get_class_info(self, rect: QRect):
        """Get classification metadata from parent widget"""
        class_id = self.classes.currentIndex()
        class_name = self.classes.currentText()
        x, y, w, h = rect.getRect()
        return {
                "category_id": class_id,
                "category_name": class_name,
                "bbox": [x, y, w, h],
                "area": w * h
            }

    def _handle_point_mode(self, click_pos):
        """Process point annotation operations"""
        if not self.targets:
            raise ValueError("No targets for point annotation")
        self.current_target = self.targets[self.current_target_id]
        if not self.current_target:
            raise ValueError("No current target for point annotation")
        
        point_info = self._get_keypoint_info(click_pos)
        self.current_target.keypoints.update(point_info)

        self._commit_changes()
    
        next_idx = (self.current_target.current_keypoint_id + 1) % self.keypoints.count()
        self.keypoints.setCurrentIndex(next_idx)
        self.visible.setCurrentIndex(2)
    
    def _on_visible_change(self, index):
        """Handle keypoint visibility change"""
        if index == 0:
            self._handle_point_mode(QPoint())
            
    def _get_keypoint_info(self, pos):
        """Get keypoint metadata from parent widget"""
        kpt_id = self.keypoints.currentIndex()
        kpt_name = self.keypoints.currentText()
        keypoint_visible = self.visible.currentIndex()
        self.current_target.current_keypoint_id = kpt_id
        return {kpt_id: {"name": kpt_name, "pos": [pos.x(), pos.y(), keypoint_visible]}}

    def _handle_bline_mode(self, click_pos):
        """Process bone line drawing operations"""
        self.current_target = self.targets[self.current_target_id]
        if not self.current_target:
            raise ValueError("No current target for bone line drawing")
            
        nearest_id = self._find_nearest_keypoint(click_pos)
        if nearest_id is None:
            return

        if not hasattr(self, '_bline_temp_points'):
            self._bline_temp_points = []
            
        self._bline_temp_points.append(nearest_id)

        if len(self._bline_temp_points) == 2:
            start_id, end_id = self._bline_temp_points
            if start_id != end_id:
                self._create_skeleton_connection(start_id, end_id)
            del self._bline_temp_points
            self._commit_changes()

    def _create_skeleton_connection(self, start_id, end_id):
        """Create a skeleton connection between two keypoints"""
        skeleton_id = len(self.current_target.skeletons)
        start_pos = self.current_target.keypoints[start_id]["pos"][0:2]
        end_pos = self.current_target.keypoints[end_id]["pos"][0:2]
        self.current_target.skeletons[skeleton_id] = {
            "skeleton": (start_id, end_id),
            "pos": [*start_pos, *end_pos]
        }
        self.addskeletons.emit([start_id, end_id])

    def _find_nearest_keypoint(self, pos):
        """Find nearest keypoint within tolerance range"""
        if not self.current_target or not self.current_target.keypoints:
            return None
            
        min_distance = self.pen_settings["radius"] * 2
        nearest_id, best_dist = None, float('inf')
        
        for kpt_id, kpt_data in self.current_target.keypoints.items():
            kpt = kpt_data["pos"][0:2]
            kpt_pos = QPoint(*kpt)
            current_dist = (kpt_pos - pos).manhattanLength()
            if current_dist < min_distance and current_dist < best_dist:
                best_dist = current_dist
                nearest_id = kpt_id
        return nearest_id

    def mouseMoveEvent(self, event):
        """Handle mouse movement events"""
        super().mouseMoveEvent(event)
        self.current_pos = event.pos()
        
        if self.drawing_mode == "rect" and self.start_position:
            self._update_temp_canvas()
        elif self.drawing_mode == "bline":
            self._process_bline_move(event.pos())

    def _process_bline_move(self, pos):
        """Update bone line preview during mouse movement"""
        old_nearest = getattr(self, '_nearest_point_id', None)
        self._nearest_point_id = self._find_nearest_keypoint(pos)
        
        if self._nearest_point_id != old_nearest or hasattr(self, '_bline_temp_points'):
            self._update_temp_canvas()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        self.current_pos = None
        super().mouseReleaseEvent(event)

    def _update_temp_canvas(self):
        """Update temporary overlay canvas"""
        self.temp_pixmap.fill(Qt.transparent)
        painter = QPainter(self.temp_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.drawing_mode == "bline":
            self._draw_bline_preview(painter)
        elif self.drawing_mode == "rect" and self.start_position:
            self._draw_rect_preview(painter, self.start_position, self.current_pos)
            
        painter.end()
        self.update_pixmap()

    def _draw_rect_preview(self, painter, start, end):
        """Draw rectangle preview with dashed lines"""
        style = self.preview_style["rect"]
        pen = QPen(style["color"], style["width"], style["style"])
        
        painter.save()
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        rect = QRect(start, end).normalized()
        if rect.width() < 3 or rect.height() < 3:
            painter.drawLine(rect.topLeft(), rect.bottomRight())
            painter.drawLine(rect.topRight(), rect.bottomLeft())
        else:
            painter.drawRect(rect)
            
        painter.restore()

    def _draw_bline_preview(self, painter):
        """Draw bone line connection preview"""
        if not self.current_target:
            raise ValueError("No current target for bone line preview")
        
        # Draw hovered point highlight
        if hasattr(self, '_nearest_point_id') and self._nearest_point_id is not None:
            kpt_data = self.current_target.keypoints[self._nearest_point_id]
            kpt = kpt_data["pos"][0:2]
            pos = QPoint(*kpt)
            painter.setBrush(QColor(255, 255, 0, 180))
            painter.drawEllipse(pos, self.pen_settings["radius"] * 1.5, 
                              self.pen_settings["radius"] * 1.5)
        
        # Draw temporary connection preview
        if hasattr(self, '_bline_temp_points') and len(self._bline_temp_points) == 1:
            start_id = self._bline_temp_points[0]
            start_pos = self.current_target.keypoints[start_id]["pos"][0:2]
            start_pos = QPoint(*start_pos)

            if hasattr(self, '_nearest_point_id') and self._nearest_point_id is not None:
                end_pos = self.current_target.keypoints[self._nearest_point_id]["pos"][0:2]
                end_pos = QPoint(*end_pos)
            else:
                end_pos = self.current_pos

            pen = QPen(QColor(100, 255, 100, 150), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(start_pos, end_pos)

    def update_pixmap(self):
        """Composite main and temp canvases for display"""
        if self.main_pixmap.size() != self.size():
            self.reset_canvases()
            
        composite = QPixmap(self.size())
        composite.fill(Qt.transparent)
        painter = QPainter(composite)
        painter.drawPixmap(0, 0, self.main_pixmap)
        painter.drawPixmap(0, 0, self.temp_pixmap)
        painter.end()
        
        self.setPixmap(composite)
        self.repaint()

    def _commit_changes(self):
        """Commit changes to main canvas"""
        self.main_pixmap.fill(Qt.transparent)
        painter = QPainter(self.main_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        self.labels_view.display_target(self.targets)
        
        for target in self.targets:
            self._draw_target(target, painter)
            
        painter.end()
        self.update_pixmap()

    def _draw_target(self, target, painter):
        """Draw complete annotation target"""
        # Draw bounding box
        painter.setBrush(Qt.NoBrush)
        if target.bounding_rect:
            painter.setPen(QPen(target.color, self.pen_settings["width"]))
            rect = QRect(*target.bounding_rect["bbox"])
            painter.drawRect(rect)
            painter = self._display_target_info(target.id, target, rect, painter)
        
        # Draw keypoints
        painter.setBrush(target.color)
        for _, kpt_data in target.keypoints.items():
            if kpt_data["pos"][2] == 0:
                continue
            keypoint = kpt_data["pos"][0:2]
            pos = QPoint(*keypoint)
            painter.drawEllipse(pos, self.pen_settings["radius"], self.pen_settings["radius"])
        
        # Draw skeletons
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(target.color, self.pen_settings["width"], 
                          Qt.SolidLine))
        for _, skeleton in target.skeletons.items():
            start = QPoint(skeleton["pos"][0], skeleton["pos"][1])
            end = QPoint(skeleton["pos"][2], skeleton["pos"][3])
            start_id = skeleton["skeleton"][0]
            end_id = skeleton["skeleton"][1]
            start_id_visible = target.keypoints[start_id]["pos"][2]
            end_id_visible = target.keypoints[end_id]["pos"][2]
            if start_id_visible == 0 or end_id_visible == 0:
                continue
            painter.drawLine(start, end)
    
    def _display_target_info(self, target_id, target, rect, painter):
        """Display target information in parent widget"""
        font = painter.font()
        font_metrics = QFontMetrics(font) 
        class_name = target.bounding_rect["category_name"]
        text = f"Target {target_id}: {class_name}"
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()
        text_rect = QRect(rect.x(), rect.y() - text_height * 1.2, text_width, text_height)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, text)
        return painter
        

    def set_drawing_mode(self, mode):
        """Set active drawing tool"""
        valid_modes = ["none", "point", "rect", "bline", "line"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid drawing mode: {mode}")
        
        if mode == "line":
            if self.current_target and self.current_target.keypoints:
                self._handle_line_mode()
            else:
                print("Create keypoints first before generating skeletons")
                mode = "none"  
            
        self.drawing_mode = mode
        self.start_position = None
        self.reset_temp_canvas()
        self.setFocus()
    
    def _handle_line_mode(self):
        """Automatically create all skeleton connections from parent's definition"""
        self.current_target = self.targets[self.current_target_id]
        if not self.current_target:
            raise ValueError("No current target for line annotation")
        
        # Clear existing skeleton connections
        self.current_target.skeletons.clear()
        
        # Create connections based on predefined skeletons
        for i in range(self.skeletons.count()):
            connection_text = self.skeletons.itemText(i)
            try:
                src, dest = map(int, connection_text.strip().split('->'))
                self._validate_and_create_connection(src, dest)
            except ValueError:
                print(f"Ignoring invalid skeleton format: {connection_text}")
        
        self._commit_changes()
    
    def _validate_and_create_connection(self, src, dest):
        """Validate and create a skeleton connection between keypoints"""
        if src == dest:
            return
            
        keypoints = self.current_target.keypoints
        if src not in keypoints or dest not in keypoints:
            print(f"Missing keypoints for connection {src}->{dest}")
            return
            
        start_pos = keypoints[src]["pos"]
        end_pos = keypoints[dest]["pos"]
        
        skeleton_id = len(self.current_target.skeletons)
        self.current_target.skeletons[skeleton_id] = {
            "skeleton": (src, dest),
            "pos": [*start_pos, *end_pos]
        }

    def clear_annotations(self):
        """Clear all annotations from canvas"""
        self.targets.clear()
        self.reset_canvases()
        self.update_pixmap()

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape and self.drawing_mode not in ["none", "point"]:
            self._cancel_current_operation()
        super().keyPressEvent(event)

    def _cancel_current_operation(self):
        """Cancel current drawing operation based on mode"""
        if self.drawing_mode == "rect":
            self._cancel_rect_drawing()
        elif self.drawing_mode == "bline":
            self._cancel_bline_drawing()
        
        self.reset_temp_canvas()
        self.update_pixmap()
    
    def _cancel_rect_drawing(self):
        """Cancel rectangle drawing operation"""
        if self.start_position is not None:
            if self.current_target and not self.current_target.bounding_rect:
                self.targets.remove(self.current_target)
                self.current_target_id = self.current_target_id - 1 if self.targets else 0
            self.start_position = None
            self.current_target = None
    
    def _cancel_bline_drawing(self):
        """Cancel bone line drawing operation"""
        if hasattr(self, '_bline_temp_points'):
            del self._bline_temp_points
    
    def _on_target_deleted(self, target_id):
        """Handle target deletion event"""
        self.targets = [t for t in self.targets if t.id != target_id]
        for new_id, target in enumerate(self.targets):
            target.id = new_id
            target.color = self._generate_color_for_id(new_id)
        
        if self.current_target_id == target_id:
             self.current_target_id = min(target_id, len(self.targets) - 1) if self.targets else None
        elif self.current_target_id > target_id:
            self.current_target_id -= 1
        self.labels_view.display_target(self.targets)
        self._commit_changes()
    
    def _on_keypoints_cleared(self, target_id):
        """Handle keypoints clear event"""
        target = next((t for t in self.targets if t.id == target_id), None)
        if target:
            target.keypoints.clear()
            target.skeletons.clear()
            self._commit_changes()

    def _on_keypoint_deleted(self, target_id, kpt_id):
        """Handle keypoint deletion event"""
        target = next((t for t in self.targets if t.id == target_id), None)
        if target and kpt_id in target.keypoints:
            del target.keypoints[kpt_id]
            skeletons_to_delete = [
                skel_id for skel_id, skel_data in target.skeletons.items()
                if kpt_id in skel_data["skeleton"]
            ]
            for skel_id in skeletons_to_delete:
                del target.skeletons[skel_id]
            self._commit_changes()
    
    def _on_skeletons_cleared(self, target_id):
        """Handle skeletons clear event"""
        print(target_id)
        target = next((t for t in self.targets if t.id == target_id), None)
        if target:
            target.skeletons.clear()
            self._commit_changes()

    def _on_skeleton_deleted(self, target_id, skeleton_id):
        """Handle skeleton deletion event"""
        target = next((t for t in self.targets if t.id == target_id), None)
        if target and skeleton_id in target.skeletons:
            del target.skeletons[skeleton_id]
            self._commit_changes()