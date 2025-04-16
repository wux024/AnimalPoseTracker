from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal, QObject
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, 
                              QListWidgetItem, QGraphicsView, QCheckBox, QLayout)
from PySide6.QtGui import (QPainter, QPixmap, QPen, QColor, QCursor,
                           QWheelEvent, QBrush, QFont, QFontMetrics)


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


class AnnotationTarget:
    """Represents a single annotated object with visual elements"""
    def __init__(self, target_id, color):
        self.id = target_id
        self.color = color          # Display color
        self.class_id = 0           # Associated class ID
        self.class_name = ""        # Class name from config
        self.bounding_rect = QRect()  # Main bounding box
        self.key_points = []        # List of keypoints (QPoint)
        self.connections = []       # List of line connections [(QPoint, QPoint)]
        self.point_radius = 5       # Radius for key points

class DrawingBoard(QLabel):
    """Main canvas for creating and managing annotations"""
    
    def __init__(self, parent=None):
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
        self.start_position = None
        self.current_pos = None
        self.color_index = 0
        
        # Visual settings
        self.pen_settings = {
            "width": 2,
            "style": Qt.SolidLine,
            "preview_alpha": 180
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

    def generate_color(self):
        """Generate distinct color using HSV cycling"""
        hue = (self.color_index * 30) % 360
        self.color_index += 1
        return QColor.fromHsv(hue, 255, 200)

    def create_target(self):
        """Create new annotation target with initial properties"""
        color = self.generate_color()
        new_target = AnnotationTarget(len(self.targets), color)
        
        # Sync class info from parent if available
        parent = self.parent()
        if parent and hasattr(parent, 'class_selector'):
            new_target.class_id = parent.class_selector.currentIndex()
            new_target.class_name = parent.class_selector.currentText()
            
        self.targets.append(new_target)
        return new_target

    def mousePressEvent(self, event):
        if self.drawing_mode == "none":
            return
            
        pos = event.pos()
        
        if self.drawing_mode == "point":
            self._handle_point_mode(pos)
        elif self.drawing_mode == "rect":
            if self.start_position is None:
                self.start_position = pos
                self.current_target = self.create_target()
                self.current_target.bounding_rect = QRect(pos, pos)
            else:
                end_pos = pos
                if (abs(self.start_position.x() - end_pos.x()) > 5 or 
                    abs(self.start_position.y() - end_pos.y()) > 5):
                    rect = QRect(self.start_position, end_pos).normalized()
                    self.current_target.bounding_rect = rect
                    self._add_to_connections()
                    self._commit_changes()
                self.start_position = None
                self.current_target = None
                self.reset_temp_canvas()
        elif self.drawing_mode == "line":
            if self.start_position is None:
                self.start_position = pos
                self.current_target = self.create_target()
                self.current_target.connections.append([pos, pos])
            else:
                end_pos = pos
                if (abs(self.start_position.x() - end_pos.x()) > 5 or 
                    abs(self.start_position.y() - end_pos.y()) > 5):
                    self.current_target.connections[-1][1] = end_pos
                    self._commit_changes()
                self.start_position = None
                self.current_target = None
                self.reset_temp_canvas()
                
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_mode == "rect" and self.start_position:
            self.current_pos = event.pos()
            self._update_temp_canvas()
        elif self.drawing_mode == "line" and self.start_position:
            self.current_pos = event.pos()
            self._update_temp_canvas()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.current_pos = None
        super().mouseReleaseEvent(event)

    def _update_temp_canvas(self):
        self.temp_pixmap.fill(Qt.transparent)
        painter = QPainter(self.temp_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.drawing_mode == "rect" and self.start_position and self.current_pos:
            self._draw_rectangle_preview(painter, self.start_position, self.current_pos)
        elif self.drawing_mode == "line" and self.start_position and self.current_pos:
            self._draw_line_preview(painter, self.start_position, self.current_pos)
            
        painter.end()
        self.update_pixmap()
    
    def _draw_rectangle_preview(self, painter, start, end):
        style = self.preview_style["rect"]
        pen = QPen(style["color"], style["width"], style["style"])
        painter.setPen(pen)
        painter.drawRect(QRect(start, end).normalized())

    def _draw_line_preview(self, painter, start, end):
        style = self.preview_style["line"]
        pen = QPen(style["color"], style["width"], style["style"])
        painter.setPen(pen)
        painter.drawLine(start, end)

    def _commit_changes(self):
        """Update main canvas with current state"""
        self.main_pixmap.fill(Qt.transparent)
        painter = QPainter(self.main_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw all existing annotations
        for target in self.targets:
            self._draw_target(target, painter)
            
        painter.end()
        self.update_pixmap()

    def update_pixmap(self):
        composite = QPixmap(self.size())
        composite.fill(Qt.transparent)
        
        painter = QPainter(composite)
        painter.drawPixmap(0, 0, self.main_pixmap) 
        painter.drawPixmap(0, 0, self.temp_pixmap)  
        painter.end()
        
        self.setPixmap(composite)
        self.repaint()


    def _handle_point_mode(self, click_pos):
        """Process point drawing operations"""
        if click_pos is None:
            return
        if not self.current_target:
            self.current_target = self.create_target()
        self.current_target.key_points.append(click_pos)
        self._commit_changes()

    def _handle_rect_mode_start(self):
        """Initialize new rectangle annotation"""
        self.current_target = self.create_target()
        self.current_target.bounding_rect = QRect(self.start_position, QSize())

    def _handle_rect_mode_end(self, end_pos):
        """Finalize rectangle dimensions"""
        if (abs(self.start_position.x() - end_pos.x()) > 5 and 
            abs(self.start_position.y() - end_pos.y()) > 5):
            self.current_target.bounding_rect = QRect(
                self.start_position, end_pos
            ).normalized()
            print("Rect:", self.current_target.bounding_rect)
            self._add_to_connections()
            self._commit_changes()

    def _handle_line_mode_start(self):
        """Initialize new line drawing"""
        self.current_target = self.create_target()
        self.current_target.connections.append([self.start_position, self.start_position])

    def _handle_line_mode_end(self, end_pos):
        """Finalize line connection"""
        if self.current_target.connections:
            self.current_target.connections[-1][1] = end_pos
            self._commit_changes()

    def _add_to_connections(self):
        """Add default connection for new rectangles"""
        if self.current_target.bounding_rect.isValid():
            center = self.current_target.bounding_rect.center()
            self.current_target.connections.append([center, center])

    def _draw_target(self, target, painter):
        """Draw complete annotation target"""
        # Draw bounding box
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(target.color, self.pen_settings["width"]))
        painter.drawRect(target.bounding_rect)
        
        # Draw key points
        painter.setBrush(target.color)
        for point in target.key_points:
            if point and hasattr(target, 'point_radius') and target.point_radius:
                painter.drawEllipse(point, target.point_radius, target.point_radius)
            else:
                print("Invalid point:", point)
        
        # Draw connections
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(target.color, self.pen_settings["width"], self.pen_settings["style"]))
        for start, end in target.connections:
            painter.drawLine(start, end)

    def set_drawing_mode(self, mode):
        """Set active drawing tool"""
        valid_modes = ["none", "point", "rect", "line"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}")
            
        self.drawing_mode = mode
        self.start_position = None
        self.current_target = None
        self.reset_temp_canvas()
        self.setFocus()

    def clear_annotations(self):
        """Remove all existing annotations"""
        self.targets.clear()
        self.reset_canvases()
        self.update_pixmap()