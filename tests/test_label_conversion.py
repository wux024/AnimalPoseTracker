"""The label converters sit between the annotator's per-image JSON and the dataset loaders.

The annotator writes ``bbox: None`` for a target that only ever got keypoints drawn on it, and
both converters used to unpack that as if it were a box.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from animalposetracker.utils.dataset import convert_labels_to_coco, convert_labels_to_yolo

PROJECT_CONFIG = {
    "keypoints_name": ["head", "tail"],
    "classes_name": ["mouse"],
    "skeleton": [[0, 1]],
}

BOXED = {
    "id": 1,
    "category_id": 0,
    "bbox": [10, 10, 20, 20],
    "area": 400,
    "keypoints": [20, 20, 1, 30, 30, 1],
}
POINT_ONLY = {
    "id": 2,
    "category_id": None,
    "bbox": None,
    "area": None,
    "keypoints": [50, 20, 1, 60, 30, 1],
}


def _write_labels(directory, annotations):
    payload = {
        "images": [{"id": 0, "file_name": "frame.png", "width": 100, "height": 50}],
        "annotations": [{"image_id": 0, **annotation} for annotation in annotations],
    }
    (Path(directory) / "frame.json").write_text(json.dumps(payload), encoding="utf-8")


class MissingBoundingBoxTests(unittest.TestCase):
    def test_coco_conversion_skips_point_only_targets_and_says_so(self):
        with tempfile.TemporaryDirectory() as temporary:
            _write_labels(temporary, [BOXED, POINT_ONLY])
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                output = convert_labels_to_coco(temporary, PROJECT_CONFIG)
            coco = json.loads(Path(output).read_text(encoding="utf-8"))

        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual(coco["annotations"][0]["category_id"], 1)
        self.assertEqual(coco["annotations"][0]["bbox"], [10, 10, 20, 20])
        # The dropped instance has to be visible somewhere, or the export looks complete.
        self.assertIn("Skipped 1 annotation(s)", captured.getvalue())

    def test_yolo_conversion_skips_point_only_targets_and_says_so(self):
        with tempfile.TemporaryDirectory() as temporary:
            _write_labels(temporary, [BOXED, POINT_ONLY])
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                output = convert_labels_to_yolo(temporary)

            lines = (Path(output) / "frame.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("0 0.200000 0.400000"), lines)
        self.assertIn("Skipped 1 annotation(s)", captured.getvalue())

    def test_boxed_targets_are_unaffected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _write_labels(temporary, [BOXED])
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                output = convert_labels_to_coco(temporary, PROJECT_CONFIG)
            coco = json.loads(Path(output).read_text(encoding="utf-8"))

        self.assertEqual(len(coco["annotations"]), 1)
        self.assertNotIn("Skipped", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
