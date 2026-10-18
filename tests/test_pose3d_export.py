"""The pose3d result exporter: flat CSV, JSON summary, and a trajectory plot."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from animalposetracker.pose3d.export import (
    iter_rows,
    load_poses,
    run,
    summarize,
    write_csv,
    write_plot,
)


def _payload():
    return {
        "schema": "pose3d/1",
        "world_unit": "mm",
        "frames": [
            {
                "frame": 0,
                "instances": [
                    {
                        "keypoints_3d": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "valid": [True, True, False],
                        "confidence": [0.9, 0.8, 0.0],
                        "reprojection_error": [0.3, 0.5, 0.0],
                        "views": ["camA", "camB", "camC"],
                    },
                    {
                        "keypoints_3d": [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]],
                        "valid": [True, True, True],
                        "confidence": [0.7, 0.7, 0.7],
                        "reprojection_error": [1.0, 1.0, 1.0],
                        "views": ["camA", "camB"],
                    },
                ],
            },
            {"frame": 1, "instances": []},
        ],
    }


class RowTests(unittest.TestCase):
    def test_valid_rows_are_flat_and_complete(self):
        rows = list(iter_rows(_payload()))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["frame"], 0)
        self.assertEqual(rows[0]["instance"], 0)
        self.assertEqual(rows[0]["keypoint"], 0)
        self.assertEqual(rows[0]["num_views"], 3)
        self.assertTrue(rows[0]["valid"])

    def test_invalid_points_are_optional(self):
        self.assertEqual(len(list(iter_rows(_payload(), include_invalid=True))), 6)

    def test_the_summary_counts_coverage_and_bounds(self):
        summary = summarize(_payload())
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["instances"], 2)
        self.assertEqual(summary["points"], 5)
        self.assertEqual(summary["dropped_points"], 1)
        self.assertAlmostEqual(summary["dropped_ratio"], 1.0 / 6.0, places=6)
        self.assertEqual(summary["reprojection_median_px"], 1.0)
        self.assertEqual(summary["keypoints"]["0"], {"valid": 2, "total": 2})
        self.assertEqual(summary["keypoints"]["2"], {"valid": 1, "total": 2})
        self.assertEqual(summary["bounds"]["min"], [0.0, 0.0, 0.0])
        self.assertEqual(summary["bounds"]["max"], [11.0, 11.0, 10.0])

    def test_a_malformed_payload_is_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "poses3d.json"
            path.write_text(json.dumps({"frames": "nope"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_poses(path)


class FileTests(unittest.TestCase):
    def test_the_csv_has_a_header_and_one_row_per_valid_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "poses3d.csv"
            self.assertEqual(write_csv(_payload(), path), 5)
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["keypoint"], "0")
        self.assertEqual(rows[0]["num_views"], "3")
        self.assertEqual(rows[-1]["instance"], "1")

    def test_the_plot_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.png"
            write_plot(_payload(), path, instance=0)
            self.assertGreater(path.stat().st_size, 0)

    def test_the_cli_writes_csv_summary_and_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poses = root / "poses3d.json"
            poses.write_text(json.dumps(_payload()), encoding="utf-8")
            output = root / "out.csv"
            summary = root / "summary.json"
            plot = root / "trajectory.png"
            self.assertEqual(run(["--poses", str(poses), "--output", str(output),
                                  "--summary-output", str(summary), "--plot", str(plot)]), 0)
            self.assertTrue(output.is_file())
            self.assertTrue(summary.is_file())
            self.assertTrue(plot.is_file())
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["points"], 5)

    def test_a_missing_payload_fails_with_a_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(run(["--poses", str(Path(temporary) / "nope.json")]), 1)

    def test_the_filtered_flag_asks_for_the_sibling_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "probe-AnimalRTPose-N-20260101"
            (project / "configs").mkdir(parents=True)
            (project / "results" / "pose3d").mkdir(parents=True)
            (project / "results" / "pose3d" / "poses3d.json").write_text(
                json.dumps(_payload()), encoding="utf-8"
            )
            self.assertEqual(run(["--config", str(project / "configs" / "runtime.yaml"), "--filtered"]), 1)


if __name__ == "__main__":
    unittest.main()
