from PySide6.QtCore import QThread, Signal, QMutex

from datetime import datetime


def measure_time(func, *args, **kwargs):
    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    return result, total_time

class BaseThread(QThread):
    """
    Base thread class that provides common attributes and methods.
    """
    status_update = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self._lock = QMutex()

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

    def _release_resources(self):
        pass