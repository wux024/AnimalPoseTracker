import cv2
import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex
import queue
import traceback


class VideoReaderThread(QThread):
    """Thread to run video reading"""
    # Signal emitted when a frame is read from file
    original_frame = Signal(np.ndarray)
    # Signal emitted to update the processing status
    status_update = Signal(str)

    def __init__(self, cap=None, parent=None):
        """
        Initialize the VideoProcessorThread.

        :param source: Video source, e.g., camera index or video file path
        :param parent: Parent object
        :param processing_function: Function to process each frame
        """
        super().__init__(parent)
        self._cap = cap
        self.running = False
        self._lock = QMutex()

    @property
    def cap(self):
        """
        Get the video source.

        :return: Video source
        """
        return self._cap

    @cap.setter
    def cap(self, value):
        """
        Set the video source.

        :param value: Video source
        """
        self._cap = value

    def run(self):
        """
        Start the video processing loop.
        """
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.status_update.emit("Read started")
            while self.is_running:
                ret, frame = self.cap.read()
                if ret:
                    self.original_frame.emit(frame.copy())
                else:
                    self.status_update.emit("Read stopped: No more frames")
                    break
                time.sleep(1.0 / fps)
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(VideoReaderThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()

    @property
    def is_running(self):
        """
        Check if the thread is running.

        :return: True if running, False otherwise
        """
        self._lock.lock()
        is_running = self.running
        self._lock.unlock()
        return is_running

    def safe_stop(self):
        """
        Safely stop the video processing thread.
        """
        self._lock.lock()
        self.running = False
        self._lock.unlock()
        self.wait()
        self._release_resources()

    def _release_resources(self):
        """
        Release the video capture resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.status_update.emit("Video capture resources released")

class VideoWriterThread(QThread):
    """Thread to run video writing"""
    # Signal emitted when a frame is written to file
    def __init__(self, save_path = None, fps=30, frame_size=(640, 480)):
        super().__init__()
        self._save_path = save_path
        self._fps = fps
        self._frame_size = frame_size
        self.video_writer = None
        self.frames = []
        self._stop_flag = False
        self._lock = QMutex()
    
    @property
    def save_path(self):
        return self._save_path
    
    @save_path.setter
    def save_path(self, value):
        self._save_path = value
    
    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        self._fps = value
    
    @property
    def frame_size(self):
        return self._frame_size
    
    @frame_size.setter
    def frame_size(self, value):
        self._frame_size = value
    
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, self.frame_size)
        while not self._stop_flag:
            if self.frames:
                frame = self.frames.pop(0)
                self.video_writer.write(frame)
        self.video_writer.release()
    
    def add_frame(self, frame):
        self.frames.append(frame)
    
    def safe_stop(self):
        """
        Safely stop the video processing thread.
        """
        self._lock.lock()
        self._stop_flag = True
        self._lock.unlock()
        self.wait()
        self._release_resources()

class FrameProcessorThread(QThread):
    """Thread to run frame processing"""
    # Signal emitted when a frame is processed
    frame_processed = Signal(np.ndarray)
    # Signal emitted to update the processing status
    status_update = Signal(str)

    def __init__(self, processing_function=None, processing_kwargs=None, buffer_queue=None, parent=None):
        """
        Initialize the FrameProcessorThread.

        :param processing_function: Function to process each frame
        :param parent: Parent object
        """
        super().__init__(parent)
        self._processing_function = processing_function
        self._processing_kwargs = processing_kwargs if processing_kwargs is not None else {}
        self.running = False
        self._lock = QMutex()
        self.buffer_queue = buffer_queue

    @property
    def processing_function(self):
        """
        Get the processing function.

        :return: Processing function
        """
        return self._processing_function

    @processing_function.setter
    def processing_function(self, value):
        """
        Set the processing function.

        :param value: Processing function
        """
        self._processing_function = value
    
    @property
    def processing_kwargs(self):
        """
        Get the processing function keyword arguments.

        :return: Processing function keyword arguments
        """
        return self._processing_kwargs

    @processing_kwargs.setter
    def processing_kwargs(self, value):
        """
        Set the processing function keyword arguments.

        :param value: Processing function keyword arguments
        """
        self._processing_kwargs = value

    def run(self):
        """
        Start the frame processing loop.
        """
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            self.status_update.emit("Frame processor started")
            while self.is_running:
                try:
                    frame = self.buffer_queue.get(timeout=1)
                    if self._processing_kwargs:
                        processed_frame = self._processing_function(frame, **self._processing_kwargs)
                    else:
                        processed_frame = self._processing_function(frame)
                    if isinstance(processed_frame, tuple):
                        self.frame_processed.emit(processed_frame[0])
                    else:
                        self.frame_processed.emit(processed_frame)
                except queue.Empty:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(FrameProcessorThread) Unexpected error: \n{error_info}")
        finally:
            self._lock.lock()
            self.running = False
            self._lock.unlock()

    @property
    def is_running(self):
        """
        Check if the thread is running.

        :return: True if running, False otherwise
        """
        self._lock.lock()
        is_running = self.running
        self._lock.unlock()
        return is_running

    def safe_stop(self):
        """
        Safely stop the frame processing thread.
        """
        self._lock.lock()
        self.running = False
        self._lock.unlock()
        self.wait()

