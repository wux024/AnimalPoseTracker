import cv2
import time
import numpy as np
from PySide6.QtCore import Signal
import queue
import traceback

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
            # A camera commonly reports 0 before it starts streaming, and `1 / 0` used to end the
            # reader after a single frame -- a preview or a run that stopped on its own.
            if not fps or fps <= 0:
                fps = 30.0
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
            self.request_stop()
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
            self.request_stop()


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
            self.request_stop()


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
                    # Guarded: with a coarse clock every stage could measure 0.0 and the inverse
                    # came out as ten million "fps" in the overlay.
                    stage_time = pre_time + infer_time + post_time
                    fps = 1.0 / stage_time if stage_time > 0 else 0.0
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
            self.request_stop()


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
            self.request_stop()

class VideoWriterThread(BaseThread):
    """Thread to run video writing"""
    def __init__(self, save_path=None, fps=30):
        super().__init__()
        self._save_path = save_path
        self._fps = fps
        self._frame_size = None
        self.video_writer = None
        self.frames = queue.Queue(maxsize=1024)
        self._stop_flag = False
        self.frame_count = 0
        self.skipped_frames = 0

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
        """Pixel size of the video being written, taken from the frames themselves."""
        return self._frame_size
    
    @property
    def stop_flag(self):
        return self._stop_flag
    
    @stop_flag.setter
    def stop_flag(self, value):
        self._stop_flag = value

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        while True:
            if self._stop_flag:
                # Frames already handed over still belong in the file: drain what is queued and
                # only then leave. Breaking straight out dropped everything the writer had not
                # consumed yet, and a short run produced no file at all.
                try:
                    frame = self.frames.get_nowait()
                except queue.Empty:
                    break
            else:
                try:
                    frame = self.frames.get(timeout=1, block=True)
                except queue.Empty:
                    continue
            # Size the writer from the frames themselves. The UI's width/height boxes are what it
            # asked the source for, and a camera is free to hand back another size -- writing
            # those frames into a writer opened at the requested size stretches all of them.
            height, width = frame.shape[:2]
            size = (width, height)
            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, size)
            elif size != self._frame_size:
                # A mid-stream size change cannot be written into an open writer; drop it and
                # keep the count so the loss is visible rather than silent.
                self.skipped_frames += 1
                continue
            self.video_writer.write(frame)
            self._frame_size = size
            self.frame_count += 1
        if self.video_writer is not None:
            self.video_writer.release()
        self.finished.emit()

    def add_frame(self, frame):
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass


    def safe_stop(self):
        """Ask the writer to finish, then wait.

        ``stop_flag`` has to be set first: the run loop only leaves through it, so waiting on a
        writer that was never told to stop blocked the caller forever -- and the caller is the
        GUI thread on End/close.
        """
        self._stop_flag = True
        super().safe_stop()
        self.frame_count = 0
        self._release_resources()

    def _release_resources(self):
        if self.video_writer is not None:
            self.video_writer.release()
