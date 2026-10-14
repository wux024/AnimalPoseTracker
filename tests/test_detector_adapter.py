"""Geometry tests for the ONNX detector adapter's output contract.

These tests do not need a real ONNX file: they build the adapter via
``object.__new__`` and plug in a fake session that returns fixed (1, N, 6)
rows, so the focus stays on the coordinate math, the class/score filtering and
the "declared conventions" (coord_mode / preprocess_mode).
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from animalposetracker.inference.detector_adapter import OnnxDetectorAdapter
from animalposetracker.inference.stream_inference import DetectorBox


class _FakeInput:
    name = "images"


class _FakeSession:
    def __init__(self, rows):
        self._rows = np.asarray(rows, dtype=np.float32)[None]  # (1, N, 6)

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, _feed):
        return [self._rows]


def _adapter(rows, *, width=100, height=50, **kwargs):
    """Build an adapter whose session returns ``rows`` and whose frames are WxH."""
    adapter = object.__new__(OnnxDetectorAdapter)
    adapter.onnx_path = Path("fake.onnx")
    adapter.input_size = kwargs.pop("input_size", 640)
    adapter.conf_threshold = kwargs.pop("conf_threshold", 0.5)
    adapter.class_ids = kwargs.pop("class_ids", (0,))
    adapter.max_det = kwargs.pop("max_det", 300)
    adapter.coord_mode = kwargs.pop("coord_mode", "normalized")
    adapter.preprocess_mode = kwargs.pop("preprocess_mode", "stretch")
    adapter.session = _FakeSession(rows)
    adapter.input_name = "images"
    return adapter, np.zeros((height, width, 3), dtype=np.uint8)


class DetectorAdapterGeometryTests(unittest.TestCase):
    def test_normalized_stretch_maps_back_to_source_frame(self):
        # One box: center (0.5, 0.5), size (0.5, 0.5), score 0.9, person.
        adapter, frame = _adapter([[0.5, 0.5, 0.5, 0.5, 0.9, 0.0]])
        boxes = adapter.predict_boxes(frame)
        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        self.assertIsInstance(box, DetectorBox)
        np.testing.assert_allclose(box.bbox_xyxy, [25, 12.5, 75, 37.5], atol=1e-4)
        self.assertAlmostEqual(box.score, 0.9, places=6)
        self.assertEqual(box.class_id, 0)

    def test_pixels_mode_skips_normalization(self):
        # Same numbers interpreted as network-input pixels (640 canvas).
        adapter, frame = _adapter(
            [[320, 160, 320, 160, 0.9, 0.0]], coord_mode="pixels"
        )
        boxes = adapter.predict_boxes(frame)
        self.assertEqual(len(boxes), 1)
        # Canvas box [160, 80, 480, 240]; stretch ratios (6.4, 12.8)
        # -> source box [25, 6.25, 75, 18.75].
        np.testing.assert_allclose(
            boxes[0].bbox_xyxy, [25.0, 6.25, 75.0, 18.75], atol=1e-4
        )

    def test_letterbox_inverse_removes_padding(self):
        # 100x50 letterboxed into 640: scale 6.4, image 640x320, pad_y (640-320)//2=160.
        adapter, frame = _adapter(
            [[0.5, 0.5, 0.5, 0.25, 0.9, 0.0]], preprocess_mode="letterbox"
        )
        boxes = adapter.predict_boxes(frame)
        self.assertEqual(len(boxes), 1)
        # canvas box [160, 240, 480, 400] -> minus pad (0, 160) -> /6.4
        # -> source box [25, 12.5, 75, 37.5] (center (50, 25), size 50x25).
        np.testing.assert_allclose(
            boxes[0].bbox_xyxy, [25.0, 12.5, 75.0, 37.5], atol=1e-3
        )

    def test_pixels_and_normalized_modes_agree_on_the_same_box(self):
        # (pixels, letterbox) is the fourth corner of the 2x2 convention matrix;
        # the same box written in both coordinate modes must decode identically.
        normalized, frame = _adapter(
            [[0.5, 0.5, 0.5, 0.25, 0.9, 0.0]], preprocess_mode="letterbox"
        )
        as_pixels, _ = _adapter(
            [[320, 320, 320, 160, 0.9, 0.0]],
            coord_mode="pixels",
            preprocess_mode="letterbox",
        )
        np.testing.assert_allclose(
            normalized.predict_boxes(frame)[0].bbox_xyxy,
            as_pixels.predict_boxes(frame)[0].bbox_xyxy,
            atol=1e-4,
        )

    def test_score_threshold_and_class_filter(self):
        rows = [
            [0.5, 0.5, 0.2, 0.2, 0.4, 0.0],  # below threshold
            [0.5, 0.5, 0.2, 0.2, 0.9, 5.0],  # wrong class
            [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],  # kept
        ]
        adapter, frame = _adapter(rows)
        boxes = adapter.predict_boxes(frame)
        self.assertEqual(len(boxes), 1)

    def test_invalid_coord_or_preprocess_mode_rejected(self):
        with patch("onnxruntime.InferenceSession"):
            with self.assertRaises(ValueError):
                OnnxDetectorAdapter(Path(__file__), coord_mode="nope")
            with self.assertRaises(ValueError):
                OnnxDetectorAdapter(Path(__file__), preprocess_mode="nope")


class DetectorProviderSelectionTests(unittest.TestCase):
    """The detector must grab the GPU whenever this onnxruntime build has one.

    A CPU-only build (like this development machine) keeps the old behaviour;
    a server with ``onnxruntime-gpu`` must not leave the detector on the CPU
    while the pose model runs on the GPU.
    """

    def test_cuda_is_preferred_when_offered(self):
        from animalposetracker.inference.detector_adapter import _default_providers

        self.assertEqual(
            _default_providers(
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            ),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_a_cpu_only_build_falls_back_to_cpu(self):
        from animalposetracker.inference.detector_adapter import _default_providers

        self.assertEqual(_default_providers(["CPUExecutionProvider"]), ["CPUExecutionProvider"])
        self.assertEqual(_default_providers([]), ["CPUExecutionProvider"])

    def test_constructor_hands_the_resolved_providers_to_ort(self):
        with patch(
            "onnxruntime.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch("onnxruntime.InferenceSession") as session:
            session.return_value.get_inputs.return_value = [_FakeInput()]
            adapter = OnnxDetectorAdapter(Path(__file__))

        self.assertEqual(
            session.call_args.kwargs["providers"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(adapter.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_explicit_providers_stay_in_charge(self):
        with patch("onnxruntime.InferenceSession") as session, patch(
            "onnxruntime.get_available_providers",
            side_effect=AssertionError("must not consult the build when told"),
        ):
            session.return_value.get_inputs.return_value = [_FakeInput()]
            adapter = OnnxDetectorAdapter(Path(__file__), providers=["CPUExecutionProvider"])

        self.assertEqual(adapter.providers, ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
