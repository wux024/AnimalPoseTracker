"""The 3D protocol: calibration validation, the opt-in project layout, and a synthetic run."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from animalposetracker.pose3d import CalibrationError, load_camera_rig, reconstruct_project


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(up, float), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ eye


def _camera_block(eye, intrinsic):
    rotation, translation = _look_at(eye)
    return {
        "image_size": [640, 480],
        "intrinsic": intrinsic.tolist(),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "distortion": {"model": "none"},
    }


def _write_project(root: Path, pose3d: bool) -> Path:
    project = root / "probe-test-AnimalRTPose-N-20260101"
    (project / "configs").mkdir(parents=True)
    (project / "results").mkdir()
    (project / "project.yaml").write_text(
        yaml.safe_dump({"project_name": "probe", "pose3d": pose3d}), encoding="utf-8"
    )
    return project


INTRINSIC = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


class CalibrationTests(unittest.TestCase):
    def _write(self, root: Path, payload) -> Path:
        path = root / "cameras.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_a_missing_file_is_reported_with_its_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(CalibrationError) as caught:
                load_camera_rig(Path(temporary) / "cameras.yaml")
        self.assertIn("cameras.yaml", str(caught.exception))

    def test_an_empty_cameras_section_asks_to_fill_it_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), {"world_unit": "mm", "cameras": {}})
            with self.assertRaises(CalibrationError) as caught:
                load_camera_rig(path)
        self.assertIn("cameras:", str(caught.exception))

    def test_a_missing_field_names_the_camera_and_the_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = _camera_block((1.0, 0.0, 0.0), INTRINSIC)
            block.pop("translation")
            path = self._write(Path(temporary), {"world_unit": "mm", "cameras": {"camA": block}})
            with self.assertRaises(CalibrationError) as caught:
                load_camera_rig(path)
        message = str(caught.exception)
        self.assertIn("camA", message)
        self.assertIn("translation", message)

    def test_fisheye_needs_four_coefficients(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = _camera_block((1.0, 0.0, 0.0), INTRINSIC)
            block["distortion"] = {"model": "fisheye", "coefficients": [0.2]}
            path = self._write(Path(temporary), {"world_unit": "mm", "cameras": {"camA": block}})
            with self.assertRaises(CalibrationError) as caught:
                load_camera_rig(path)
        self.assertIn("4 coefficients", str(caught.exception))

    def test_a_valid_rig_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = _camera_block((1.0, 0.5, 0.3), INTRINSIC)
            path = self._write(Path(temporary), {"world_unit": "cm", "cameras": {"camA": block}})
            rig = load_camera_rig(path)
        self.assertEqual(list(rig.cameras), ["camA"])
        self.assertEqual(rig.world_unit, "cm")
        self.assertEqual(rig.distortion["camA"][0], "none")


class ReconstructionTests(unittest.TestCase):
    def _fake_predictions(self, skeleton, cameras, directory: Path, frames: int = 4):
        for name, eye in cameras.items():
            rotation, translation = _look_at(eye)
            records = []
            for frame in range(frames):
                points = skeleton + frame * 0.01
                camera_points = (rotation @ points.T).T + translation
                pixels = (INTRINSIC @ camera_points.T).T
                pixels = pixels[:, :2] / pixels[:, 2:3]
                keypoints = [[float(u), float(v), 1.0] for u, v in pixels]
                records.append({"source": f"frame{frame}.png", "predictions": [{"keypoints": keypoints}]})
            target = directory / "results" / "predict" / name
            target.mkdir(parents=True)
            (target / "predictions.json").write_text(json.dumps(records), encoding="utf-8")

    def test_a_pose3d_project_triangulates_its_predictions(self):
        rng = np.random.default_rng(3)
        skeleton = rng.normal(scale=0.1, size=(5, 3))
        cameras = {"camA": (1.2, 0.0, 0.3), "camB": (-0.9, 0.8, 0.3), "camC": (-0.5, -1.0, 0.2)}
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=True)
            payload = {
                "world_unit": "mm",
                "cameras": {name: _camera_block(eye, INTRINSIC) for name, eye in cameras.items()},
            }
            (project / "configs" / "cameras.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
            self._fake_predictions(skeleton, cameras, project)

            report = reconstruct_project(project)

            poses = json.loads((project / "results" / "pose3d" / "poses3d.json").read_text(encoding="utf-8"))
            recovered = np.asarray(poses["frames"][0]["instances"][0]["keypoints_3d"], dtype=np.float64)
            self.assertEqual(report["summary"]["frames"], 4)
            self.assertEqual(report["summary"]["instances"], 4)
            # the file rounds to 1e-4 of a world unit, so that is the precision of the check
            self.assertLess(float(np.abs(recovered - skeleton).max()), 1e-3)
            self.assertTrue((project / "results" / "pose3d" / "report.json").is_file())
            self.assertFalse((project / "results" / "pose3d" / "poses3d-filter.json").exists())
            self.assertEqual(report["calibration"]["snapshot"]["cameras"]["camA"]["image_size"], [640, 480])

    def test_filtering_writes_a_sibling_file(self):
        skeleton = np.zeros((3, 3))
        cameras = {"camA": (1.0, 0.0, 0.0), "camB": (-1.0, 0.0, 0.0)}
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=True)
            payload = {
                "world_unit": "mm",
                "cameras": {name: _camera_block(eye, INTRINSIC) for name, eye in cameras.items()},
            }
            (project / "configs" / "cameras.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
            self._fake_predictions(skeleton, cameras, project, frames=6)
            report = reconstruct_project(project, pose_filter={"method": "median", "window_length": 5})
        self.assertIsNotNone(report["outputs"]["poses3d_filtered"])

    def test_a_project_without_the_3d_option_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=False)
            with self.assertRaises(ValueError) as caught:
                reconstruct_project(project)
        self.assertIn("--pose3d", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
