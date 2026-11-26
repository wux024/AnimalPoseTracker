"""SimCC validator must report stage timings and image counts alongside the pose metrics."""

import math
import unittest

import numpy as np
import torch

from animalposetracker.evaluation.topdown import SimCCPoseValidator


class _FakeAnnotationIndex:
    """The validator reads ``validation_dataset.annotations`` as the COCO annotation index."""

    annotation_path = None
    image_paths = []
    image_id_by_path = {}
    category_id_by_class = {}
    data_config = {}
    num_keypoints = 2
    source_class_names = ["chimpanzee"]
    skeleton = []


class _FakeDataset:
    """Minimal stand-in: no annotation file, no images, so GT building stays in memory."""

    def __init__(self):
        self.annotations = _FakeAnnotationIndex()


class _FakeModel:
    def __init__(self, num_keypoints: int = 2, length: int = 64, peak: int = 16):
        self.num_keypoints = num_keypoints
        self.length = length
        self.peak = peak
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        batch = images.shape[0]
        pred_x = torch.zeros((batch, self.num_keypoints, self.length), dtype=torch.float32)
        pred_y = torch.zeros((batch, self.num_keypoints, self.length), dtype=torch.float32)
        pred_x[..., self.peak] = 1.0
        pred_y[..., self.peak] = 1.0
        return pred_x, pred_y


def _make_batch(num_keypoints: int = 2, peak: int = 16, split_ratio: float = 2.0):
    coordinate = peak / split_ratio
    targets = {
        "warp_inverse": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
        ),
        "keypoints": torch.full(
            (1, num_keypoints, 2), float(coordinate), dtype=torch.float32
        ),
        "keypoints_visible": torch.ones((1, num_keypoints), dtype=torch.float32),
        "bbox_xyxy": torch.tensor([[0.0, 0.0, 16.0, 16.0]], dtype=torch.float32),
        "area": torch.tensor([256.0], dtype=torch.float32),
        "image_id": torch.tensor([1], dtype=torch.long),
        "category_id": torch.tensor([0], dtype=torch.long),
    }
    images = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
    return {"images": images, "targets": targets}


def _make_validator(num_keypoints: int = 2):
    return SimCCPoseValidator(
        input_size=(32, 32),
        split_ratio=2.0,
        flip_indices=None,
        kpt_oks_sigmas=[0.05] * num_keypoints,
        validation_dataset=_FakeDataset(),
        flip_tta=False,
        ap_metric="YOLO/AP",
        oks_area_mode="bbox",
    )


class SimCCTimingTests(unittest.TestCase):
    def test_validator_reports_speed_images_and_instances(self):
        validator = _make_validator()
        model = _FakeModel()
        metrics = validator(model, [_make_batch()], torch.device("cpu"))

        for key in (
            "speed/preprocess_ms",
            "speed/inference_ms",
            "speed/postprocess_ms",
            "images",
            "instances",
        ):
            self.assertIn(key, metrics, key)
            self.assertTrue(math.isfinite(metrics[key]), key)
            self.assertGreaterEqual(metrics[key], 0.0, key)

        self.assertEqual(metrics["images"], 1.0)
        self.assertEqual(metrics["instances"], 1.0)
        self.assertEqual(model.calls, 1)

    def test_speed_is_per_instance_not_per_batch(self):
        validator = _make_validator()
        model = _FakeModel()
        metrics = validator(model, [_make_batch(), _make_batch()], torch.device("cpu"))

        # Two batches of one crop each -> two instances, two distinct source images only if the
        # ids differ; both batches carry image_id 1, so images stays at one.
        self.assertEqual(metrics["instances"], 2.0)
        self.assertEqual(metrics["images"], 1.0)
        # Per-instance averaging keeps the value finite and small for toy tensors.
        self.assertLess(metrics["speed/inference_ms"], 1000.0)


if __name__ == "__main__":
    unittest.main()
