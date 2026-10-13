"""The shared NMS core: engines, similarity and the named threshold defaults.

Every engine case pins the exact output the pre-refactor implementations produced for a
fixed input (captured before the move), so a behavioural drift fails here first instead of
somewhere downstream in the decode/inference tests.
"""

import unittest

import numpy as np


class NmsCoreEngineTests(unittest.TestCase):
    """Bit-for-bit guards for the two index engines and the OKS expression."""

    def test_numpy_engine_matches_the_pre_refactor_output(self):
        from animalposetracker.postprocessing.nms_core import nms_indices_numpy

        boxes = np.array(
            [[10, 10, 50, 50], [11, 11, 51, 51], [100, 100, 140, 140], [101, 101, 141, 141]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        class_ids = np.array([0, 0, 1, 1], dtype=np.int64)
        kw = dict(confidence_threshold=0.25, iou_threshold=0.45)

        self.assertEqual(nms_indices_numpy(boxes, scores, class_ids, agnostic=True, **kw).tolist(), [0, 2])
        self.assertEqual(nms_indices_numpy(boxes, scores, class_ids, agnostic=False, **kw).tolist(), [0, 2])

        tie_boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 240, 240]], dtype=np.float32)
        tie_scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        tie_classes = np.array([0, 0, 0], dtype=np.int64)
        self.assertEqual(
            nms_indices_numpy(tie_boxes, tie_scores, tie_classes, agnostic=True, **kw).tolist(), [2, 0]
        )

    def test_numpy_engine_handles_empty_input(self):
        from animalposetracker.postprocessing.nms_core import nms_indices_numpy

        empty = nms_indices_numpy(
            np.empty((0, 4), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int64),
            agnostic=True,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        self.assertEqual(empty.dtype, np.int64)
        self.assertEqual(len(empty), 0)

    def test_torch_engine_matches_the_pre_refactor_output(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - torch ships with the training extra
            self.skipTest("torch is not installed")
        from animalposetracker.postprocessing.nms_core import nms_indices_torch

        boxes = torch.tensor(
            [[10, 10, 50, 50], [11, 11, 51, 51], [100, 100, 140, 140], [101, 101, 141, 141]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        self.assertEqual(nms_indices_torch(boxes, scores, 0.5).tolist(), [0, 2])

        chain_boxes = torch.tensor(
            [[0, 0, 100, 100], [5, 5, 105, 105], [10, 10, 110, 110], [500, 500, 600, 600]],
            dtype=torch.float32,
        )
        chain_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        self.assertEqual(nms_indices_torch(chain_boxes, chain_scores, 0.3).tolist(), [0, 3])

        empty = nms_indices_torch(torch.empty((0, 4)), torch.empty((0,)), 0.5)
        self.assertEqual(empty.dtype, torch.int64)
        self.assertEqual(len(empty), 0)

    def test_oks_similarity_matches_the_inline_expression(self):
        from animalposetracker.postprocessing.nms_core import oks_similarity

        kp_a = np.array([[10, 10, 1], [20, 20, 1], [30, 30, 1]], dtype=np.float32)
        kp_b = np.array([[10.5, 10.5, 1], [20.5, 20.5, 1], [30.5, 30.5, 1]], dtype=np.float32)
        variances = (np.ones(3, dtype=np.float32) * 2.0) ** 2

        delta = kp_b[:, :2] - kp_a[:, :2]
        denominator = variances * ((900.0 + 900.0) * 0.5 + np.spacing(1)) * 2.0
        expected = float(np.exp(-np.sum(delta * delta, axis=1) / denominator).mean())

        self.assertEqual(oks_similarity(kp_a, kp_b, 900.0, 900.0, variances), expected)

    def test_oks_nms_keeps_what_the_previous_loop_kept(self):
        from animalposetracker.evaluation.topdown import SimCCPoseValidator

        class Fake:
            kpt_oks_sigmas = np.ones(3, dtype=np.float32)

        def det(kps, score):
            return {"image_id": 1, "category_id": 1, "keypoints": kps, "score": score, "area": 900.0}

        kps_a = [[10, 10, 1], [20, 20, 1], [30, 30, 1]]
        kps_b = [[10.5, 10.5, 1], [20.5, 20.5, 1], [30.5, 30.5, 1]]
        kps_c = [[100, 100, 1], [110, 110, 1], [120, 120, 1]]

        kept = SimCCPoseValidator._oks_nms(
            Fake(), [det(kps_a, 0.9), det(kps_b, 0.8), det(kps_c, 0.7)], 0.5
        )
        self.assertEqual([d["score"] for d in kept], [0.9, 0.7])  # b suppressed by a

        untouched = SimCCPoseValidator._oks_nms(Fake(), [det(kps_a, 0.9), det(kps_c, 0.7)], 0.5)
        self.assertEqual([d["score"] for d in untouched], [0.9, 0.7])  # far keypoints, nothing dropped

        self.assertEqual(SimCCPoseValidator._oks_nms(Fake(), [], 0.5), [])


class NmsThresholdConstantTests(unittest.TestCase):
    """The defaults stop being scattered literals: each one references its named constant."""

    def test_signature_defaults_reference_the_named_constants(self):
        import dataclasses
        import inspect

        from animalposetracker.postprocessing import nms_params as params
        from animalposetracker.postprocessing.nms import pose_non_max_suppression
        from animalposetracker.postprocessing.output_decoders import PoseOutputConfig

        signature = inspect.signature(pose_non_max_suppression)
        self.assertEqual(signature.parameters["confidence_threshold"].default, params.TRAINING_NMS_CONF)
        self.assertEqual(signature.parameters["iou_threshold"].default, params.TRAINING_NMS_IOU)
        self.assertEqual(signature.parameters["max_detections"].default, params.TRAINING_NMS_MAX_DET)
        self.assertIs(signature.parameters["agnostic"].default, params.TRAINING_NMS_AGNOSTIC)

        fields = {field.name: field.default for field in dataclasses.fields(PoseOutputConfig)}
        self.assertEqual(fields["confidence_threshold"], params.DEPLOY_NMS_CONF)
        self.assertEqual(fields["iou_threshold"], params.DEPLOY_NMS_IOU)
        self.assertIs(fields["class_agnostic_nms"], params.DEPLOY_NMS_AGNOSTIC)

    def test_session_builder_defaults_reference_the_named_constants(self):
        import inspect

        from animalposetracker.inference.session import build_inference_session
        from animalposetracker.postprocessing import nms_params as params

        signature = inspect.signature(build_inference_session)
        self.assertEqual(signature.parameters["conf"].default, params.DEPLOY_NMS_CONF)
        self.assertEqual(signature.parameters["iou"].default, params.DEPLOY_NMS_IOU)
        self.assertIs(signature.parameters["agnostic_nms"].default, params.DEPLOY_NMS_AGNOSTIC)


class InferenceEngineSurfaceTests(unittest.TestCase):
    """The retired one-frame helpers are gone; the live surface must stay importable."""

    def test_inference_engine_no_longer_carries_the_retired_helpers(self):
        from animalposetracker.inference.inferencer import InferenceEngine

        for name in ("postprocess", "warmup", "process_frame"):
            self.assertFalse(hasattr(InferenceEngine, name), name)

    def test_the_live_surface_still_imports(self):
        from animalposetracker.inference.session import build_inference_session  # noqa: F401
        from animalposetracker.inference.stream_inference import StreamInferencePipeline  # noqa: F401


class LetterboxCoreTests(unittest.TestCase):
    """The shared geometry stays the value both letterbox adapters act on."""

    def test_geometry_matches_both_adapters(self):
        import numpy as np

        from animalposetracker.preprocessing.letterbox import letterbox_image
        from animalposetracker.preprocessing.letterbox_core import (
            PAD_VALUE,
            letterbox_padding,
            letterbox_resized_size,
            letterbox_scale,
        )
        from animalposetracker.preprocessing.live import preprocess_live_frame

        image = np.full((5, 7, 3), 9, dtype=np.uint8)

        canvas, scale, left, top = letterbox_image(image, 10, 8)
        shared_scale = letterbox_scale(7, 5, 10, 8)
        self.assertEqual(scale, shared_scale)
        resized_w, resized_h = letterbox_resized_size(7, 5, shared_scale)
        self.assertEqual((left, top), letterbox_padding(10, 8, resized_w, resized_h))
        # This geometry pads only the bottom: the corner pixel is image content (9), and the
        # first row below the pasted image is the pad (114).
        self.assertEqual(int(canvas[0, 0, 0]), 9)
        self.assertEqual(int(canvas[top + resized_h, 0, 0]), PAD_VALUE)

        _out, inverse = preprocess_live_frame(image, 10, 8)
        self.assertAlmostEqual(float(np.asarray(inverse)[0][0]), 1.0 / shared_scale, places=5)


if __name__ == "__main__":
    unittest.main()
