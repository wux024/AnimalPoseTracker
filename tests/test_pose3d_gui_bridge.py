"""The 3D page builds CLI commands; this checks they match the CLI's own flags."""

import tempfile
import unittest
from pathlib import Path

import yaml

from animalposetracker.pose3d.gui_bridge import (
    build_calibrate_command,
    build_export_command,
    build_pose3d_command,
    camera_names,
    default_views,
    prediction_path,
)


class CameraListTests(unittest.TestCase):
    def _write(self, root: Path, payload) -> Path:
        (root / "configs").mkdir(parents=True, exist_ok=True)
        path = root / "configs" / "cameras.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_camera_names_follow_the_file_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, {"cameras": {"left": {}, "right": {}, "top": {}}})
            self.assertEqual(camera_names(root), ["left", "right", "top"])

    def test_a_missing_or_broken_file_yields_no_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(camera_names(root), [])
            (root / "configs").mkdir()
            (root / "configs" / "cameras.yaml").write_text("cameras: [unclosed", encoding="utf-8")
            self.assertEqual(camera_names(root), [])

    def test_an_empty_template_has_no_cameras_yet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, {"cameras": {}})
            self.assertEqual(camera_names(root), [])

    def test_default_views_point_at_each_camera_prediction_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root, {"cameras": {"camA": {}, "camB": {}}})
            views = default_views(root)
            self.assertEqual(views["camA"], root / "results" / "predict" / "camA" / "predictions.json")
            self.assertEqual(prediction_path(root, "camA", filtered=True).name,
                             "predictions-filter.json")


class CalibrateCommandTests(unittest.TestCase):
    def test_every_field_reaches_the_command(self):
        command = build_calibrate_command(
            "/tmp/project",
            {"camA": "/data/camA", "camB": "/data/camB"},
            board_squares="10x7", square_size=22.0, marker_size=16.5,
            sync="software", force=True, intrinsics_from="/tmp/old.yaml",
        )
        self.assertEqual(command[1:4], ["-m", "animalposetracker.cli", "calibrate"])
        self.assertEqual(command[command.index("--view") + 1], f"camA={Path('/data/camA')}")
        self.assertEqual(sum(1 for item in command if item == "--view"), 2)
        self.assertEqual(command[command.index("--board-squares") + 1], "10x7")
        self.assertEqual(command[command.index("--square-size") + 1], "22")
        self.assertEqual(command[command.index("--marker-size") + 1], "16.5")
        self.assertEqual(command[command.index("--sync") + 1], "software")
        self.assertIn("--force", command)
        self.assertEqual(command[command.index("--intrinsics-from") + 1], str(Path("/tmp/old.yaml")))

    def test_force_is_opt_in(self):
        command = build_calibrate_command("/tmp/p", {"a": "/a", "b": "/b"})
        self.assertNotIn("--force", command)

    def test_one_view_is_refused(self):
        with self.assertRaises(ValueError):
            build_calibrate_command("/tmp/p", {"camA": "/data/camA"})

    def test_an_empty_path_is_refused(self):
        with self.assertRaises(ValueError):
            build_calibrate_command("/tmp/p", {"camA": "/data/camA", "camB": "  "})

    def test_no_views_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            build_calibrate_command("/tmp/p", {})


class ReconstructionCommandTests(unittest.TestCase):
    def test_defaults_match_the_cli_defaults(self):
        command = build_pose3d_command("/tmp/project")
        self.assertEqual(command[3], "pose3d")
        self.assertEqual(command[command.index("--config") + 1], "configs/runtime.yaml")
        self.assertEqual(command[command.index("--association") + 1], "index")
        self.assertEqual(command[command.index("--max-sampson") + 1], "25")
        self.assertNotIn("--check-sync", command)

    def test_epipolar_options_and_explicit_views(self):
        command = build_pose3d_command(
            "/tmp/project", views={"camA": "/tmp/a.json", "camB": "/tmp/b.json"},
            association="epipolar", max_sampson=18.0, min_views=3, check_sync=True,
            pose_filter="median", filter_window=7,
        )
        self.assertEqual(command[command.index("--association") + 1], "epipolar")
        self.assertEqual(command[command.index("--max-sampson") + 1], "18")
        self.assertEqual(command[command.index("--min-views") + 1], "3")
        self.assertIn("--check-sync", command)
        self.assertEqual(command[command.index("--pose-filter") + 1], "median")
        self.assertEqual(command[command.index("--filter-window") + 1], "7")
        self.assertEqual(sum(1 for item in command if item == "--views"), 2)


class ExportCommandTests(unittest.TestCase):
    def test_filtered_and_plot_options(self):
        command = build_export_command("/tmp/project", filtered=True,
                                      plot="/tmp/trajectory.png", instance=2)
        self.assertEqual(command[3], "pose3d-render")
        self.assertIn("--filtered", command)
        self.assertEqual(command[command.index("--plot") + 1], str(Path("/tmp/trajectory.png")))
        self.assertEqual(command[command.index("--instance") + 1], "2")

    def test_plain_export_has_no_optional_flags(self):
        command = build_export_command("/tmp/project")
        for flag in ("--filtered", "--plot", "--include-invalid", "--video"):
            self.assertNotIn(flag, command)

    def test_rendering_rides_on_the_export_command(self):
        command = build_export_command("/tmp/project", filtered=True, video="both")
        self.assertEqual(command[3], "pose3d-render")
        self.assertEqual(command[command.index("--video") + 1], "both")
        self.assertIn("--filtered", command)


if __name__ == "__main__":
    unittest.main()
