import cv2
import time
import numpy as np
from PySide6.QtCore import Signal
import queue
import traceback
import json
from pathlib import Path

from .base import BaseThread, measure_time

class VideoReaderThread(BaseThread):
    """Thread to run video reading"""
    # frame
    data_ready = Signal(np.ndarray)

    def __init__(self, parent=None, cap=None):
        super().__init__(parent)
        self._cap = cap

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
                try:
                    ret, frame = self._cap.read()
                    if not ret:
                        self.status_update.emit("Read finished")
                        self.finished.emit()
                        break
                    self.data_ready.emit(frame.copy())
                    time.sleep(1/fps)
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(VideoReaderThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()
            self._release_resources()

    def _release_resources(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.status_update.emit("Video capture resources released")


class PreprocessThread(BaseThread):
    """Thread to run preprocess"""
    # frame, img, IM, pre_time
    data_ready = Signal(np.ndarray, np.ndarray, np.ndarray, float)

    def __init__(self, parent=None, 
                 preprocess_function=None, 
                 input_queue=None
                 ):
        super().__init__(parent)
        self._preprocess_function = preprocess_function
        self.input_queue = input_queue

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
                    frame = self.input_queue.get(timeout=1, block=False)
                    [img, IM], pre_time = measure_time(self._preprocess_function, frame)
                    self.data_ready.emit(frame, img, IM, pre_time)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(PreprocessThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()


class InferenceThread(BaseThread):
    """Thread to run inference"""
    # frame, pred, IM, pre_time, infer_time
    data_ready = Signal(np.ndarray, np.ndarray, np.ndarray, float, float)
    def __init__(self, parent=None,
                 inference_function=None,
                 input_queue=None):
        super().__init__(parent)
        self._inference_function = inference_function
        self.input_queue = input_queue

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
                    frame, img, IM, pre_time = self.input_queue.get(timeout=1, block=True)
                    pred, infer_time = measure_time(self._inference_function, img)
                    self.data_ready.emit(frame, pred, IM, pre_time, infer_time)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(InferenceThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()


class PostprocessThread(BaseThread):
    """Thread to run postprocessing"""
    # frame, results
    data_ready = Signal(np.ndarray, dict)
    def __init__(self, parent=None,
                 postprocess_function=None,
                 input_queue=None):
        super().__init__(parent)
        self._postprocess_function = postprocess_function
        self.input_queue = input_queue

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
                    frame, pred, IM, pre_time, infer_time = self.input_queue.get(timeout=1, block=True)
                    results, post_time = measure_time(self._postprocess_function, pred, IM)
                    fps = 1.0 / (pre_time + infer_time + post_time + 1e-7)
                    results.update(
                        {'preprocess_time': pre_time,
                         'inference_time': infer_time,
                         'postprocess_time': post_time,
                         'fps': fps})
                    self.data_ready.emit(frame, results)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(PostprocessThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()


class VisualizeThread(BaseThread):
    """Thread to run visualization"""
    # frame, results
    data_ready = Signal(np.ndarray, dict)
    finished_checked = Signal()
    def __init__(self, parent=None,
                 visualize_function=None,
                 input_queue=None):
        super().__init__(parent)
        self._visualize_function = visualize_function
        self.input_queue = input_queue

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
                    frame, results = self.input_queue.get(timeout=1, block=True)
                    visualized_frame = self._visualize_function(frame, 
                                                                results)
                    self.data_ready.emit(visualized_frame, results)
                except queue.Empty:
                    self.finished_checked.emit()
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(VisualizeThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()

class TrackerThread(BaseThread):
    """Thread to run tracking"""
    # frame, results
    data_ready = Signal(np.ndarray, dict)
    def __init__(self, parent=None,
                 tracker_function=None,
                 input_queue=None):
        super().__init__(parent)
        self._tracker_function = tracker_function
        self.input_queue = input_queue

    @property
    def tracker_function(self):
        return self._tracker_function

    @tracker_function.setter
    def tracker_function(self, value):
        self._tracker_function = value

    def run(self):
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            while self.is_running:
                try:
                    frame, results = self.input_queue.get(timeout=1, block=True)
                    tracked_results = self._tracker_function(results)
                    self.data_ready.emit(frame, tracked_results)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(TrackerThread) Unexpected error: \n{error_info}")
        finally:
            self.safe_stop()    

class VideoWriterThread(BaseThread):
    """Thread to run video writing"""
    def __init__(self, save_path=None, fps=30, frame_size=(640, 480)):
        super().__init__()
        self._save_path = save_path
        self._fps = fps
        self._frame_size = frame_size
        self.video_writer = None
        self.frames = []
        self.results = []
        self._stop_flag = False
        self.frame_count = 0

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
                results = self.results.pop(0)
                self.save_results(results)
                self.frame_count += 1
        self.video_writer.release()

    def add_frame(self, frame, results):
        self.frames.append(frame)
        self.results.append(results)
    
    def save_results(self, results):
        save_dir = Path(self.save_path).parent / f"frame_{self.frame_count}.json"
        with open(save_dir, 'w') as f:
            json.dump(results, f, indent=4)

    def safe_stop(self):
        super().safe_stop()
        self.frame_count = 0
        self._release_resources()

    def _release_resources(self):
        if self.video_writer is not None:
            self.video_writer.release()
