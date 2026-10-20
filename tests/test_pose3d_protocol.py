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

    def test_the_batched_backend_recovers_the_same_skeleton(self):
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

            report = reconstruct_project(project, backend="numpy")

            poses = json.loads((project / "results" / "pose3d" / "poses3d.json").read_text(encoding="utf-8"))
            recovered = np.asarray(poses["frames"][0]["instances"][0]["keypoints_3d"], dtype=np.float64)
            self.assertLess(float(np.abs(recovered - skeleton).max()), 1e-3)
            self.assertEqual(report["triangulation"],
                             {"backend": "numpy", "refine": True, "reject_outliers": True})

    def test_the_polished_and_unpolished_runs_agree(self):
        rng = np.random.default_rng(5)
        skeleton = rng.normal(scale=0.1, size=(5, 3))
        cameras = {"camA": (1.2, 0.0, 0.3), "camB": (-0.9, 0.8, 0.3), "camC": (-0.5, -1.0, 0.2)}
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=True)
            payload = {
                "world_unit": "mm",
                "cameras": {name: _camera_block(eye, INTRINSIC) for name, eye in cameras.items()},
            }
            (project / "configs" / "cameras.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
            self._fake_predictions(skeleton, cameras, project, frames=6)

            reconstruct_project(project, output_dir=project / "polished")
            reconstruct_project(project, output_dir=project / "raw", refine=False)

            def load(name):
                path = project / name / "poses3d.json"
                return json.loads(path.read_text(encoding="utf-8"))["frames"]

            for polished, raw in zip(load("polished"), load("raw")):
                for a, b in zip(polished["instances"], raw["instances"]):
                    np.testing.assert_allclose(a["keypoints_3d"], b["keypoints_3d"], atol=1e-3)
                    self.assertEqual(a["valid"], b["valid"])

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

    def test_a_public_benchmark_project_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=True)
            identity = yaml.safe_load((project / "project.yaml").read_text(encoding="utf-8"))
            identity["public_dataset"] = "Tri-Mouse"
            (project / "project.yaml").write_text(yaml.safe_dump(identity), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                reconstruct_project(project)
        self.assertIn("public benchmark", str(caught.exception))

    def test_the_create_command_refuses_pose3d_for_a_packaged_dataset(self):
        import subprocess
        import sys

        from animalposetracker.cfg import DATA_YAML_PATHS

        packaged = next(iter(DATA_YAML_PATHS.values()))
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-m", "animalposetracker.cli", "create",
                 "--dataset", str(packaged), "--dataset-root", temporary,
                 "--workspace", temporary, "--name", "public3d", "--worker", "t",
                 "--model", "AnimalRTPose", "--scale", "N", "--pose3d"],
                capture_output=True, text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("public benchmark", completed.stderr + completed.stdout)

    def test_video_frame_records_are_accepted_and_sorted(self):
        skeleton = np.zeros((3, 3))
        cameras = {"camA": (1.0, 0.0, 0.0), "camB": (-1.0, 0.0, 0.0)}
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=True)
            payload = {
                "world_unit": "mm",
                "cameras": {name: _camera_block(eye, INTRINSIC) for name, eye in cameras.items()},
            }
            (project / "configs" / "cameras.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
            for name in cameras:
                target = project / "results" / "predict" / name
                target.mkdir(parents=True)
                rotation, translation = _look_at(cameras[name])
                points = (rotation @ skeleton.T).T + translation
                pixels = (INTRINSIC @ points.T).T
                pixels = pixels[:, :2] / pixels[:, 2:3]
                keypoints = [[float(u), float(v), 1.0] for u, v in pixels]
                # deliberately out of order, with a per-video summary record appended
                records = [
                    {"source": "cam.mp4", "frame": 1, "predictions": [{"keypoints": keypoints}]},
                    {"source": "cam.mp4", "frame": 0, "predictions": [{"keypoints": keypoints}]},
                    {"source": "cam.mp4", "output": "rendered.mp4", "frames": 2},
                ]
                (target / "predictions.json").write_text(json.dumps(records), encoding="utf-8")
            report = reconstruct_project(project)
        self.assertEqual(report["summary"]["frames"], 2)
        self.assertEqual(report["summary"]["instances"], 2)

    def test_a_project_without_the_3d_option_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary), pose3d=False)
            with self.assertRaises(ValueError) as caught:
                reconstruct_project(project)
        self.assertIn("--pose3d", str(caught.exception))


class SyncTests(unittest.TestCase):
    def _rig(self):
        from animalposetracker.pose3d import CameraCalibration, CameraRig
        cameras = {name: CameraCalibration(name, INTRINSIC, *_look_at(eye))
                   for name, eye in (("camA", (1.2, 0.0, 0.3)), ("camB", (-1.0, 0.6, 0.3)))}
        return CameraRig(cameras=cameras, distortion={n: ("none", np.zeros(4)) for n in cameras})

    @staticmethod
    def _project(points, camera):
        local = points @ camera.rotation_matrix.T + camera.translation_vector
        pixels = (local / local[:, 2:3]) @ camera.intrinsic_matrix.T
        return pixels[:, :2]

    def _series(self, rig):
        rng = np.random.default_rng(11)
        base = rng.normal(scale=0.1, size=(6, 3))
        series = {"camA": [], "camB": []}
        for frame in range(30):
            points = base + 0.02 * frame + 0.05 * np.sin(np.arange(6))[:, None] * frame / 30
            for name, camera in rig.cameras.items():
                series[name].append(np.hstack([self._project(points, camera), np.ones((6, 1))]))
        return series

    def test_a_frame_offset_is_recovered_from_the_epipolar_constraint(self):
        from animalposetracker.pose3d.sync import estimate_frame_offsets

        rig = self._rig()
        series = self._series(rig)
        shifted = {"camA": series["camA"],
                   "camB": [series["camB"][0]] * 3 + series["camB"]}
        report = estimate_frame_offsets(rig, shifted, max_offset=8)
        self.assertEqual(report["offsets"]["camB"], 3)
        self.assertTrue(report["curves"]["camB"]["confident"])

    def test_the_sync_block_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = _camera_block((1.0, 0.0, 0.0), INTRINSIC)
            payload = {"world_unit": "mm", "cameras": {"camA": block},
                       "sync": {"mode": "telepathy"}}
            path = Path(temporary) / "cameras.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(CalibrationError) as caught:
                load_camera_rig(path)
        self.assertIn("sync.mode", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
