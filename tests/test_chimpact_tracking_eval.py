"""The ChimpACT tracking evaluation ships its own self-check; keep it green."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "chimpact_tracking_eval.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("chimpact_tracking_eval", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChimpactTrackingEvalTests(unittest.TestCase):
    def test_self_check_passes(self):
        self.assertEqual(load_tool()._self_check(), 0)

    def test_idf1_is_one_when_every_gt_is_matched_once(self):
        module = load_tool()
        idf1, idp, idr, idtp = module.idf1_metrics({("a", 1): 5, ("b", 2): 5}, 10, 10)

        self.assertAlmostEqual(idf1, 1.0)
        self.assertAlmostEqual(idp, 1.0)
        self.assertAlmostEqual(idr, 1.0)
        self.assertEqual(idtp, 10)

    def test_labels_map_to_every_tenth_video_frame(self):
        module = load_tool()
        payload = {
            "images": [
                {"id": 0, "file_name": "clip/images/000000.jpg"},
                {"id": 1, "file_name": "clip/images/000001.jpg"},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "bbox": [0, 0, 10, 10], "bbox_id_ori": "u1", "iscrowd": 0},
                {"id": 1, "image_id": 1, "bbox": [5, 5, 10, 10], "bbox_id_ori": "u1", "iscrowd": 0},
                {"id": 2, "image_id": 1, "bbox": [5, 5, 10, 10], "bbox_id_ori": "u2", "iscrowd": 1},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)

        try:
            frames = module.load_clip_labels(path)
        finally:
            path.unlink()

        # Label index 1 belongs to video frame 10; the crowd annotation is skipped.
        self.assertEqual(sorted(frames), [0, 10])
        self.assertEqual(len(frames[0]), 1)
        self.assertEqual(len(frames[10]), 1)
        self.assertEqual(frames[0][0][0], "u1")
        self.assertEqual(frames[0][0][1], (0.0, 0.0, 10.0, 10.0))

    def test_natural_key_orders_numeric_frames(self):
        module = load_tool()
        names = ["000010.jpg", "000002.jpg", "000001.jpg"]
        self.assertEqual(
            sorted(names, key=module.natural_key),
            ["000001.jpg", "000002.jpg", "000010.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
