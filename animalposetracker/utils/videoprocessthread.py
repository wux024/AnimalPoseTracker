import cv2
import time
import numpy as np
from PySide6.QtCore import Signal
import queue
import traceback

from .base import BaseThread, FrameInfo

class VideoReaderThread(BaseThread):
    """Thread to run video reading"""
    preview_frame = Signal(np.ndarray)

    def __init__(self, parent=None, 
                 cap=None, 
                 output_queue=None,
                 preview=False):
        super().__init__(parent)
        self._cap = cap
        self._preview = preview
        self.output_queue = output_queue

    @property
    def cap(self):
        return self._cap

    @cap.setter
    def cap(self, value):
        self._cap = value
    
    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = value

    def run(self):
        try:
            self._lock.lock()
            self.running = True
            self._lock.unlock()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.status_update.emit("Read started")
            while self.is_running:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.status_update.emit("Read finished")
                        self.finished.emit()
                        break
                    if self._preview:
                        self.preview_frame.emit(frame)
                    else:
                        self.output_queue.put(frame, block=False)
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

    def __init__(self, parent=None, 
                 preprocess_function=None, 
                 input_queue=None, 
                 output_queue=None
                 ):
        super().__init__(parent)
        self._preprocess_function = preprocess_function
        self.input_queue = input_queue
        self.output_queue = output_queue

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
                    start_time = time.time()
                    img, IM = self._preprocess_function(frame)
                    pre_time = time.time() - start_time
                    self.output_queue.put((frame, img, IM, pre_time), block=False)
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

    def __init__(self, parent=None,
                 inference_function=None,
                 input_queue=None,
                 output_queue=None):
        super().__init__(parent)
        self._inference_function = inference_function
        self.input_queue = input_queue
        self.output_queue = output_queue

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
                    start_time = time.time()
                    pred = self._inference_function(img)
                    infer_time = time.time() - start_time
                    self.output_queue.put((frame, pred, IM, pre_time, infer_time), block=False)
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

    def __init__(self, parent=None,
                 postprocess_function=None,
                 input_queue=None,
                 output_queue=None):
        super().__init__(parent)
        self._postprocess_function = postprocess_function
        self.input_queue = input_queue
        self.output_queue = output_queue

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
                    start_time = time.time()
                    results = self._postprocess_function(pred, IM)
                    post_time = time.time() - start_time
                    fps = 1.0 / (pre_time + infer_time + post_time + 1e-7)
                    results.update(
                        {'preprocess_time': pre_time,
                         'inference_time': infer_time,
                         'postprocess_time': post_time,
                         'fps': fps})
                    self.output_queue.put((frame, results), block=False)
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
    visualized_frame = Signal(np.ndarray)
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
                    self.visualized_frame.emit(visualized_frame)
                except queue.Empty:
                    self.finished_checked.emit()
                except queue.Full:
                    pass
        except Exception as e:
            error_info = traceback.format_exc()
            self.status_update.emit(f"(VisualizeThread) Unexpected error: \n{error_info}")
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

    def safe_stop(self):
        super().safe_stop()
        self._release_resources()

    def _release_resources(self):
        if self.video_writer is not None:
            self.video_writer.release()
