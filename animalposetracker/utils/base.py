from PySide6.QtCore import QThread, Signal, QMutex

from time import perf_counter


def measure_time(func, *args, **kwargs):
    """Time one call of ``func``.

    ``perf_counter`` rather than ``datetime.now()``: the wall clock's tick is coarse enough
    that a fast stage measured exactly 0.0, and the postprocess thread then divided by it.
    """
    start_time = perf_counter()
    result = func(*args, **kwargs)
    return result, perf_counter() - start_time

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

    def request_stop(self):
        """Ask the run loop to finish, without waiting for it.

        Safe to call from inside ``run()``: a thread cannot wait on itself, so the
        workers' own ``finally`` blocks use this instead of :meth:`safe_stop`.
        """
        self._lock.lock()
        self.running = False
        self._lock.unlock()

    def safe_stop(self):
        """Ask the run loop to finish and wait for the thread to leave ``run()``."""
        self.request_stop()
        # Qt warns "Thread tried to wait on itself" and the wait is meaningless when this is
        # called from the thread's own run(); the guard keeps that path harmless.
        if QThread.currentThread() is not self:
            self.wait()

    def _release_resources(self):
        pass