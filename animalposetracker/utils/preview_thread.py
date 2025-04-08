from PySide6.QtCore import Signal, QThread, QMutex
import cv2
import numpy as np
import time


class PreviewThread(QThread):
    """Thread to run inference on video"""
    frame_ready = Signal(np.ndarray)
    status_update = Signal(str)
    def __init__(self, source=None, parent=None):
        super().__init__(parent)
        self.source = source
        self.running = False
        self._lock = QMutex()
    
    def run(self):
        self._lock.lock()
        self.running = True
        self._lock.unlock()
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened():
            self.status_update.emit("Preview failed to start")
            return
        self.status_update.emit("Preview started")
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.status_update.emit("Preview stopped")
                self.finished.emit()
                break
            time.sleep(1.0/fps)

        cap.release()
        self.status_update.emit("Preview stopped")
    
    @property
    def is_running(self):
        self._lock.lock()
        is_running = self.running
        self._lock.unlock()
        return is_running
    
    def safe_stop(self):
        self._lock.lock()
        self.running = False
        self._lock.unlock()
        self.wait()