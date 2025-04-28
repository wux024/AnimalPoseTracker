import cv2
import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex
import queue
import traceback
from pathlib import Path
import json


class VideoReaderThread(QThread):
    """Thread to run video reading"""
    original_frame = Signal(np.ndarray)
    status_update = Signal(str)

    def __init__(self, cap=None, parent=None):
        super().__init__(parent)
        self._cap = cap
        self.running = False
        self._lock = QMutex()

    @property
    def cap(self):
        return self._cap

    @cap.setter
    def cap(self, value):
        self._cap = value

    def run(self):
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
        self._lock.lock()
        is_running = self.running
        self._lock.unlock()
        return is_running

    def safe_stop(self):
        self._lock.lock()
        self.running = False
        self._lock.unlock()
        self.wait()
        self._release_resources()

    def _release_resources(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.status_update.emit("Video capture resources released")


class PreprocessThread(QThread):
    """Thread to run preprocess"""
    # preprocessed_frame: original frame, preprocessed image, affine matrix, preprocess time
    preprocessed_frame = Signal(np.ndarray, np.ndarray, np.ndarray, float)
    status_update = Signal(str)

    def __init__(self, preprocess_function=None, buffer_queue=None, parent=None):
        super().__init__(parent)
        self._preprocess_function = preprocess_function
        self.running = False
        self._lock = QMutex()
        self.buffer_queue = buffer_queue

    @property
    def preprocess_function(self):
        return self._preprocess_function

    @preprocess_function.setter
    def preprocess_function(self, value):
        self._preprocess_function = value

    def run(self):
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            while self.is_running:
                try:
                    frame = self.buffer_queue.get(timeout=1)
                    start_time = time.time()
                    img, IM = self._preprocess_function(frame)
                    pre_time = time.time() - start_time
                    self.preprocessed_frame.emit(frame, img, IM, pre_time)
                except queue.Empty:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(PreprocessThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()

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


class InferenceThread(QThread):
    """Thread to run inference"""
    # inference_done: original frame, predicted results, affine matrix, 
    # preprocess time, inference time
    inference_done = Signal(np.ndarray, np.ndarray, np.ndarray, float, float)
    status_update = Signal(str)

    def __init__(self, inference_function=None, buffer_queue=None, parent=None):
        super().__init__(parent)
        self.running = False
        self._lock = QMutex()
        self.buffer_queue = buffer_queue
        self._inference_function = inference_function

    @property
    def inference_function(self):
        return self._inference_function

    @inference_function.setter
    def inference_function(self, value):
        self._inference_function = value

    def run(self):
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            while self.is_running:
                try:
                    frame, img, IM, pre_time = self.buffer_queue.get(timeout=1)
                    start_time = time.time()
                    pred = self._inference_function(img)
                    infer_time = time.time() - start_time
                    self.inference_done.emit(frame, pred, IM, pre_time, infer_time)
                except queue.Empty:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(InferenceThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()

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


class PostprocessThread(QThread):
    """Thread to run postprocessing"""
    # postprocessed_frame: original frame, predicted results
    postprocessed_frame = Signal(np.ndarray, dict)
    status_update = Signal(str)

    def __init__(self, postprocess_function=None, buffer_queue=None, parent=None):
        super().__init__(parent)
        self._postprocess_function = postprocess_function
        self.running = False
        self._lock = QMutex()
        self.buffer_queue = buffer_queue

    @property
    def postprocess_function(self):
        return self._postprocess_function

    @postprocess_function.setter
    def postprocess_function(self, value):
        self._postprocess_function = value

    def run(self):
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            while self.is_running:
                try:
                    frame, pred, IM, pre_time, infer_time = self.buffer_queue.get(timeout=1)
                    start_time = time.time()
                    results = self._postprocess_function(pred, IM)
                    post_time = time.time() - start_time
                    results.update(
                        {'preprocess_time': pre_time, 
                         'inference_time': infer_time, 
                         'postprecess_time': post_time,
                         'fps': 1.0 / (pre_time + infer_time + post_time + 1e-7)})
                    self.postprocessed_frame.emit(frame, results)
                except queue.Empty:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(PostprocessThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()

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


class VisualizeThread(QThread):
    """Thread to run visualization"""
    # visualized_frame: visualized frame, predicted results
    visualized_frame = Signal(np.ndarray, dict)
    status_update = Signal(str)

    def __init__(self, visualize_function=None, buffer_queue=None, parent=None):
        super().__init__(parent)
        self._visualize_function = visualize_function
        self.running = False
        self._lock = QMutex()
        self.buffer_queue = buffer_queue

    @property
    def visualize_function(self):
        return self._visualize_function

    @visualize_function.setter
    def visualize_function(self, value):
        self._visualize_function = value

    def run(self):
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            while self.is_running:
                try:
                    frame, results = self.buffer_queue.get(timeout=1)
                    visualized_frame = self._visualize_function(frame, results)
                    self.visualized_frame.emit(visualized_frame, results)
                except queue.Empty:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(VisualizeThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()

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


class VideoWriterThread(QThread):
    """Thread to run video writing"""
    def __init__(self, save_path=None, fps=30, frame_size=(640, 480)):
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
        self._lock.lock()
        self._stop_flag = True
        self._lock.unlock()
        self.wait()
        self._release_resources()

    def _release_resources(self):
        if self.video_writer is not None:
            self.video_writer.release()
            