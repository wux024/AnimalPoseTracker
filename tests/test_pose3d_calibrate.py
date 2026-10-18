"""The calibrate command: detection, intrinsics, joint refinement, and YAML output."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from animalposetracker.pose3d import load_camera_rig
from animalposetracker.pose3d.bundle import _rodrigues
from animalposetracker.pose3d.calibrate import (
    BoardSpec,
    IntrinsicFit,
    View,
    _parse_views,
    calibrate_detections,
    detect_views,
    fit_intrinsics,
    run,
    write_camera_rig,
)

INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (640, 480)


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(up, dtype=np.float64), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ eye


def _scene(frames=8, noise=0.0, seed=0):
    spec = BoardSpec()
    points = spec.points()
    rng = np.random.default_rng(seed)
    eyes = {"camA": (1000.0, 0.0, 250.0), "camB": (-500.0, 900.0, 250.0), "camC": (-500.0, -900.0, 250.0)}
    truth = {name: _look_at(eye) for name, eye in eyes.items()}
    detections = {name: {} for name in truth}
    for frame in range(frames):
        rotation = _rodrigues([0.3 * np.sin(frame * 1.1), 0.2 * np.cos(frame * 0.8), 0.08 * frame])
        translation = np.array([10.0 * np.sin(frame), 8.0 * np.cos(frame * 1.2), 0.0])
        world = points @ rotation.T + translation
        for name, (camera_rotation, camera_translation) in truth.items():
            local = world @ camera_rotation.T + camera_translation
            pixels = (local @ INTRINSIC.T)[:, :2] / local[:, 2:3]
            assert (local[:, 2] > 0).all()
            if noise:
                pixels = pixels + rng.normal(scale=noise, size=pixels.shape)
            detections[name][frame] = (np.arange(len(points)), pixels)
    return spec, points, truth, detections


def _intrinsics():
    return {name: IntrinsicFit(INTRINSIC, "none", np.zeros(0)) for name in ("camA", "camB", "camC")}


def _sizes():
    return {name: SIZE for name in ("camA", "camB", "camC")}


def _rotation_error(left, right):
    return float(np.degrees(np.arccos(np.clip((np.trace(left @ right.T) - 1.0) / 2.0, -1.0, 1.0))))


def _anchor_frame(truth, anchor="camA"):
    """Express the synthetic extrinsics in the anchor camera's world frame."""
    anchor_rotation, anchor_translation = truth[anchor]
    result = {anchor: (np.eye(3), np.zeros(3))}
    for name, (rotation, translation) in truth.items():
        if name == anchor:
            continue
        relative = rotation @ anchor_rotation.T
        result[name] = (relative, translation - relative @ anchor_translation)
    return result


class CalibrationTests(unittest.TestCase):
    def test_noise_free_detections_recover_the_rig(self):
        spec, points, truth, detections = _scene()
        result = calibrate_detections(
            detections, _intrinsics(), _sizes(), spec, anchor="camA", holdout=0.25,
        )
        self.assertLess(result.metrics["board_holdout"]["median_mm"], 0.01)
        self.assertLess(result.metrics["board_all"]["reprojection_median_px"], 0.001)
        expected = _anchor_frame(truth)
        for name in ("camB", "camC"):
            self.assertLess(_rotation_error(result.extrinsics[name][0], expected[name][0]), 0.01)
            self.assertLess(float(np.linalg.norm(result.extrinsics[name][1] - expected[name][1])), 0.05)
        np.testing.assert_allclose(result.extrinsics["camA"][0], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result.extrinsics["camA"][1], np.zeros(3), atol=1e-12)

    def test_skip_joint_uses_the_pnp_median(self):
        spec, points, truth, detections = _scene()
        result = calibrate_detections(
            detections, _intrinsics(), _sizes(), spec, anchor="camA", skip_joint=True,
        )
        self.assertFalse(result.metrics["joint"])
        self.assertIn("board_all", result.metrics)
        expected = _anchor_frame(truth)
        for name in ("camB", "camC"):
            self.assertLess(_rotation_error(result.extrinsics[name][0], expected[name][0]), 0.01)
            self.assertLess(float(np.linalg.norm(result.extrinsics[name][1] - expected[name][1])), 0.05)

    def test_noise_still_produces_a_sane_rig(self):
        spec, points, truth, detections = _scene(noise=0.3, seed=3)
        result = calibrate_detections(
            detections, _intrinsics(), _sizes(), spec, anchor="camA", holdout=0.25,
        )
        self.assertGreater(result.metrics["board_holdout"]["median_mm"], 0.0)
        self.assertLess(result.metrics["board_all"]["reprojection_median_px"], 1.0)
        expected = _anchor_frame(truth)
        for name in ("camB", "camC"):
            self.assertLess(_rotation_error(result.extrinsics[name][0], expected[name][0]), 0.2)

    def test_a_lone_anchor_frame_cannot_be_split(self):
        spec, points, truth, detections = _scene(frames=1)
        self.assertIsNotNone(spec)
        with self.assertRaises(ValueError):
            calibrate_detections(detections, _intrinsics(), _sizes(), spec, anchor="camA")


class IntrinsicTests(unittest.TestCase):
    def test_a_pinhole_fit_recovers_the_focal_length(self):
        spec, points, truth, detections = _scene()
        fit = fit_intrinsics(detections["camA"], SIZE, spec, model="none")
        self.assertEqual(fit.model, "none")
        self.assertLess(abs(float(fit.matrix[0, 0]) - 800.0), 5.0)
        self.assertLess(fit.rms, 0.5)

    def test_too_few_frames_are_rejected(self):
        spec, points, truth, detections = _scene(frames=2)
        with self.assertRaises(ValueError):
            fit_intrinsics(detections["camA"], SIZE, spec)

    def test_an_unknown_model_is_rejected(self):
        spec, points, truth, detections = _scene()
        with self.assertRaises(ValueError):
            fit_intrinsics(detections["camA"], SIZE, spec, model="magic")


class DetectionTests(unittest.TestCase):
    def test_generated_board_images_are_detected_in_folders(self):
        spec = BoardSpec()
        image = spec.board().generateImage((600, 420), marginSize=10, borderBits=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            views = []
            for name in ("camA", "camB", "camC"):
                folder = root / name
                folder.mkdir()
                for frame in range(4):
                    cv2.imwrite(str(folder / f"{frame:03d}.png"), image)
                views.append(View(name, folder))
            detections, sizes = detect_views(views, spec)
        self.assertEqual(list(detections), ["camA", "camB", "camC"])
        for name in detections:
            self.assertEqual(len(detections[name]), 4)
            self.assertEqual(sizes[name], (600, 420))

    def test_view_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            _parse_views(["camA=frames"])
        with self.assertRaises(ValueError):
            _parse_views(["camA=frames", "camA=other"])
        with self.assertRaises(ValueError):
            _parse_views(["broken"])


class WriteTests(unittest.TestCase):
    def test_the_yaml_round_trips_and_force_is_required(self):
        spec, points, truth, detections = _scene()
        result = calibrate_detections(
            detections, _intrinsics(), _sizes(), spec, anchor="camA", holdout=0.25,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "configs" / "cameras.yaml"
            write_camera_rig(path, result)
            rig = load_camera_rig(path)
            self.assertEqual(list(rig.cameras), ["camA", "camB", "camC"])
            self.assertEqual(rig.sync_mode, "hardware")
            self.assertEqual(rig.distortion["camA"][0], "none")
            with self.assertRaises(FileExistsError):
                write_camera_rig(path, result)
            write_camera_rig(path, result, force=True)


class CliTests(unittest.TestCase):
    def test_view_is_required(self):
        with self.assertRaises(SystemExit):
            run([])

    def test_missing_input_paths_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                run(["--view", f"camA={root / 'a'}", "--view", f"camB={root / 'b'}"]), 1
            )


if __name__ == "__main__":
    unittest.main()
