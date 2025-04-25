import cv2
import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex


class VideoProcessorThread(QThread):
    """Thread to run video processing and preview"""
    # Signal emitted when a frame (raw or processed) is ready
    frame_ready = Signal(np.ndarray)
    # Signal emitted to update the processing status
    status_update = Signal(str)

    def __init__(self, source=None, parent=None, processing_function=None):
        """
        Initialize the VideoProcessorThread.

        :param source: Video source, e.g., camera index or video file path
        :param parent: Parent object
        :param processing_function: Function to process each frame
        """
        super().__init__(parent)
        self._source = source
        self.running = False
        self._lock = QMutex()
        # Frame processing function, default is None
        self._processing_function = processing_function
        self.cap = None

    @property
    def source(self):
        """
        Get the video source.

        :return: Video source
        """
        return self._source

    @source.setter
    def source(self, value):
        """
        Set the video source.

        :param value: Video source
        """
        self._source = value

    @property
    def processing_function(self):
        """
        Get the frame processing function.

        :return: Frame processing function
        """
        return self._processing_function

    @processing_function.setter
    def processing_function(self, value):
        """
        Set the frame processing function.

        :param value: Frame processing function
        """
        self._processing_function = value

    def run(self):
        """
        Start the video processing loop.
        """
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            self.cap = cv2.VideoCapture(self._source)
            if not self.cap.isOpened():
                self.status_update.emit("Preview failed to start")
                return
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.status_update.emit("Preview started")
            while self.is_running:
                ret, frame = self.cap.read()
                if ret:
                    if self._processing_function:
                        try:
                            # Process the frame
                            processed_frame = self._processing_function(frame)
                            self.frame_ready.emit(processed_frame)
                        except Exception as e:
                            self.status_update.emit(f"Processing error: {str(e)}")
                    else:
                        self.frame_ready.emit(frame)
                else:
                    self.status_update.emit("Preview stopped: No more frames")
                    break
                time.sleep(1.0 / fps)
        except Exception as e:
            self.status_update.emit(f"Unexpected error: {str(e)}")
        finally:
            self._release_resources()

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
    def __init__(self, save_path, fps=30, frame_size=(640, 480)):
        super().__init__()
        self._save_path = save_path
        self._fps = fps
        self._frame_size = frame_size
        self.video_writer = None
        self.frames = []
        self._stop_flag = False
    
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
    
    def stop(self):
        self._stop_flag = True
