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
                        if not self._preview:
                            # Send end signal
                            end_frame = FrameInfo(end=True)
                            self.output_queue.put(end_frame, block=False)
                        break
                    if self._preview:
                        self.preview_frame.emit(frame)
                    else:
                        frame_info = FrameInfo(original_frame=frame.copy())
                        self.output_queue.put(frame_info, block=False)
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
                    frame_info = self.input_queue.get(timeout=1, block=False)
                    if frame_info.end:
                        break
                    start_time = time.time()
                    img, IM = self._preprocess_function(frame_info.original_frame)
                    pre_time = time.time() - start_time
                    frame_info.preprocess_frame = img
                    frame_info.IM = IM
                    frame_info.results.update({'preprocess_time': pre_time})
                    self.output_queue.put(frame_info, block=False)
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
                    frame_info = self.input_queue.get(timeout=1, block=True)
                    if frame_info.end:
                        break
                    start_time = time.time()
                    pred = self._inference_function(frame_info.preprocess_frame)
                    infer_time = time.time() - start_time
                    frame_info.results.update({'pred': pred})
                    frame_info.results.update({'inference_time': infer_time})
                    self.output_queue.put(frame_info, block=False)
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
                    frame_info = self.input_queue.get(timeout=1, block=True)
                    if frame_info.end:
                        break
                    start_time = time.time()
                    results = self._postprocess_function(frame_info.results['pred'], frame_info.IM)
                    post_time = time.time() - start_time
                    fps = 1.0 / (frame_info.results['preprocess_time'] + frame_info.results['inference_time'] + post_time + 1e-7)
                    frame_info.results.update(
                        {'boxes': results['boxes'],
                         'keypoints_list': results['keypoints_list'],
                         'class_ids': results['class_ids'],
                         'scores': results['scores'],
                         'postprocess_time': post_time,
                         'fps': fps})
                    self.output_queue.put(frame_info, block=False)
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
    #visualized_frame = Signal(np.ndarray, dict)
    visualized_frame = Signal(np.ndarray)

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
                    frame_info = self.input_queue.get(timeout=1, block=True)
                    if frame_info.end:
                        self.finished.emit()
                        break
                    visualized_frame = self._visualize_function(frame_info.original_frame, 
                                                                frame_info.results)
                    # self.visualized_frame.emit(visualized_frame.copy(), 
                    #                            frame_info.results.copy())
                    self.visualized_frame.emit(visualized_frame.copy())
                except queue.Empty:
                    pass
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
