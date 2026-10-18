"""The joint board calibration: projection, pose algebra, and recovery tests."""

import unittest

import cv2
import numpy as np

from animalposetracker.pose3d.bundle import (
    BundleObservation,
    _build_sparsity,
    _project,
    _rodrigues,
    _rotation_vector,
    bundle_adjust,
)
from animalposetracker.pose3d.core import CameraCalibration

INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])


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


def _board_points():
    xs = np.linspace(-100.0, 100.0, 6)
    ys = np.linspace(-60.0, 60.0, 4)
    return np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)


def _scene(frames=6, noise=0.0, seed=0):
    """Three cameras around a small board with a different board pose per frame."""
    rng = np.random.default_rng(seed)
    board = _board_points()
    eyes = {"camA": (1000.0, 0.0, 200.0), "camB": (-500.0, 900.0, 200.0), "camC": (-500.0, -900.0, 200.0)}
    extrinsics = {name: _look_at(eye) for name, eye in eyes.items()}
    poses = {}
    for index in range(frames):
        tilt = 0.35 * np.sin(index * 1.1)
        vector = np.array([tilt * 0.9, 0.25 * np.cos(index * 0.7), 0.1 * index])
        rotation = _rodrigues(vector)
        translation = np.array([15.0 * np.sin(index), 10.0 * np.cos(index * 1.3), 0.0])
        poses[index] = (rotation, translation)
    cameras = {
        name: CameraCalibration(name, INTRINSIC, rotation, translation)
        for name, (rotation, translation) in extrinsics.items()
    }
    observations = []
    for frame, (board_rotation, board_translation) in poses.items():
        world = board @ board_rotation.T + board_translation
        for name, (rotation, translation) in extrinsics.items():
            pixels, depth = _project(world, rotation, translation, INTRINSIC)
            assert (depth > 0).all()
            if noise:
                pixels = pixels + rng.normal(scale=noise, size=pixels.shape)
            observations.append(BundleObservation(frame, name, np.arange(len(board)), pixels))
    return board, cameras, extrinsics, poses, observations


def _perturb(pose, rng, rotation=0.005, translation=1.0):
    rotation_matrix, translation_vector = pose
    jitter = _rodrigues(rng.normal(scale=rotation, size=3))
    return jitter @ rotation_matrix, translation_vector + rng.normal(scale=translation, size=3)


def _initial(extrinsics, poses, rng, rotation=0.005, translation=1.0):
    initial_extrinsics = {name: _perturb(pose, rng, rotation, translation) for name, pose in extrinsics.items()}
    initial_extrinsics["camA"] = extrinsics["camA"]
    initial_poses = {frame: _perturb(pose, rng, rotation, translation) for frame, pose in poses.items()}
    return initial_extrinsics, initial_poses


def _rotation_error(left, right):
    return float(np.degrees(np.arccos(np.clip((np.trace(left @ right.T) - 1.0) / 2.0, -1.0, 1.0))))


class GeometryTests(unittest.TestCase):
    def test_rodrigues_matches_opencv(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            vector = rng.normal(scale=0.8, size=3)
            self.assertLess(float(np.abs(_rodrigues(vector) - cv2.Rodrigues(vector.reshape(3, 1))[0]).max()), 1e-12)
            np.testing.assert_allclose(_rotation_vector(_rodrigues(vector)), vector, atol=1e-10)

    def test_projection_matches_hand_values_and_opencv(self):
        points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        pixels, depth = _project(points, np.eye(3), np.zeros(3), INTRINSIC)
        np.testing.assert_allclose(pixels, [[320.0, 240.0], [1120.0, 240.0], [320.0, 1040.0]])
        np.testing.assert_allclose(depth, [1.0, 1.0, 1.0])
        rng = np.random.default_rng(3)
        rotation = _rodrigues(rng.normal(scale=0.5, size=3))
        translation = np.array([100.0, -40.0, 900.0])
        projected, _ = _project(points, rotation, translation, INTRINSIC)
        expected, _ = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], translation, INTRINSIC, None)
        np.testing.assert_allclose(projected, expected.reshape(-1, 2), atol=1e-9)


class RecoveryTests(unittest.TestCase):
    def test_noise_free_state_has_no_residual(self):
        board, cameras, extrinsics, poses, observations = _scene(frames=3)
        result = bundle_adjust(
            board, observations, cameras, extrinsics, poses, anchor="camA",
            loss="linear", max_nfev=1, max_initial_residual_px=1e-3,
        )
        self.assertLess(result.median_px, 1e-9)

    def test_noise_free_recovery_hits_the_truth(self):
        board, cameras, extrinsics, poses, observations = _scene(frames=6)
        rng = np.random.default_rng(11)
        initial_extrinsics, initial_poses = _initial(extrinsics, poses, rng)
        result = bundle_adjust(
            board, observations, cameras, initial_extrinsics, initial_poses,
            anchor="camA", loss="linear", max_nfev=200, max_initial_residual_px=1e6,
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.median_px, 1e-6)
        for name in ("camB", "camC"):
            rotation_error = _rotation_error(result.extrinsics[name][0], extrinsics[name][0])
            self.assertLess(rotation_error, 1e-5)
            self.assertLess(float(np.linalg.norm(result.extrinsics[name][1] - extrinsics[name][1])), 1e-3)
        for frame, (rotation, translation) in poses.items():
            self.assertLess(_rotation_error(result.board_poses[frame][0], rotation), 1e-5)
            self.assertLess(float(np.linalg.norm(result.board_poses[frame][1] - translation)), 1e-3)

    def test_noise_is_averaged_and_the_result_beats_the_initial_state(self):
        board, cameras, extrinsics, poses, observations = _scene(frames=10, noise=0.3, seed=5)
        rng = np.random.default_rng(17)
        initial_extrinsics, initial_poses = _initial(extrinsics, poses, rng, rotation=0.02, translation=5.0)
        result = bundle_adjust(
            board, observations, cameras, initial_extrinsics, initial_poses,
            anchor="camA", max_nfev=400, max_initial_residual_px=1e6,
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.median_px, 1.0)
        for name in ("camB", "camC"):
            before = _rotation_error(initial_extrinsics[name][0], extrinsics[name][0])
            after = _rotation_error(result.extrinsics[name][0], extrinsics[name][0])
            self.assertLess(after, 0.2 * before)
            self.assertLess(after, 0.1)
            self.assertLess(
                float(np.linalg.norm(result.extrinsics[name][1] - extrinsics[name][1])), 5.0
            )

    def test_an_outlier_frame_is_downweighted(self):
        board, cameras, extrinsics, poses, observations = _scene(frames=8)
        clean = bundle_adjust(
            board, observations, cameras, extrinsics, poses, anchor="camA",
            max_nfev=200, max_initial_residual_px=1e6,
        )
        corrupted = []
        for observation in observations:
            pixels = observation.pixels
            if observation.frame == 3 and observation.camera == "camA":
                pixels = pixels + 100.0
            corrupted.append(BundleObservation(observation.frame, observation.camera,
                                               observation.corner_ids, pixels))
        result = bundle_adjust(
            board, corrupted, cameras, extrinsics, poses, anchor="camA",
            max_nfev=200, max_initial_residual_px=1e6,
        )
        self.assertTrue(result.success, result.message)
        for name in ("camB", "camC"):
            self.assertLess(_rotation_error(result.extrinsics[name][0], clean.extrinsics[name][0]), 0.05)
            self.assertLess(
                float(np.linalg.norm(result.extrinsics[name][1] - clean.extrinsics[name][1])), 1.0
            )


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.board, self.cameras, self.extrinsics, self.poses, self.observations = _scene(frames=4)
        rng = np.random.default_rng(23)
        self.initial_extrinsics, self.initial_poses = _initial(self.extrinsics, self.poses, rng)

    def _run(self, **overrides):
        arguments = dict(
            board_points=self.board, observations=self.observations, cameras=self.cameras,
            initial_extrinsics=self.initial_extrinsics, initial_board_poses=self.initial_poses,
            anchor="camA", max_nfev=20, max_initial_residual_px=1e6,
        )
        arguments.update(overrides)
        return bundle_adjust(**arguments)

    def test_the_anchor_camera_is_not_optimized(self):
        result = self._run()
        np.testing.assert_array_equal(result.extrinsics["camA"][0], self.initial_extrinsics["camA"][0])
        np.testing.assert_array_equal(result.extrinsics["camA"][1], self.initial_extrinsics["camA"][1])

    def test_the_freeze_switches_keep_their_blocks_fixed(self):
        board_only = self._run(optimize_extrinsics=False)
        for name in self.initial_extrinsics:
            np.testing.assert_allclose(board_only.extrinsics[name][0], self.initial_extrinsics[name][0], atol=1e-12)
            np.testing.assert_allclose(board_only.extrinsics[name][1], self.initial_extrinsics[name][1], atol=1e-12)
        extrinsics_only = self._run(optimize_board=False)
        for frame in self.initial_poses:
            np.testing.assert_allclose(
                extrinsics_only.board_poses[frame][0], self.initial_poses[frame][0], atol=1e-12
            )
            np.testing.assert_allclose(
                extrinsics_only.board_poses[frame][1], self.initial_poses[frame][1], atol=1e-12
            )

    def test_the_initial_residual_gate_names_the_problem(self):
        self.initial_poses = {
            frame: (rotation, translation + 300.0)
            for frame, (rotation, translation) in self.initial_poses.items()
        }
        with self.assertRaises(ValueError) as caught:
            self._run(max_initial_residual_px=10.0)
        self.assertIn("initial residual", str(caught.exception))

    def test_an_unknown_camera_is_named(self):
        self.observations[0] = BundleObservation(0, "camZ", np.arange(4), self.observations[0].pixels[:4])
        with self.assertRaises(ValueError) as caught:
            self._run()
        self.assertIn("camZ", str(caught.exception))

    def test_a_corner_id_outside_the_board_is_rejected(self):
        count = len(self.board)
        self.observations[0] = BundleObservation(
            0, "camA", np.arange(count + 1), np.zeros((count + 1, 2))
        )
        with self.assertRaises(ValueError) as caught:
            self._run()
        self.assertIn("outside", str(caught.exception))

    def test_duplicate_frame_camera_pairs_are_rejected(self):
        self.observations.append(self.observations[0])
        with self.assertRaises(ValueError) as caught:
            self._run()
        self.assertIn("duplicate", str(caught.exception))

    def test_non_finite_pixels_are_rejected(self):
        pixels = self.observations[0].pixels.copy()
        pixels[0, 0] = np.nan
        self.observations[0] = BundleObservation(0, "camA", np.arange(len(pixels)), pixels)
        with self.assertRaises(ValueError) as caught:
            self._run()
        self.assertIn("finite", str(caught.exception))

    def test_a_missing_board_pose_is_rejected(self):
        self.initial_poses.pop(max(self.initial_poses))
        with self.assertRaises(ValueError) as caught:
            self._run()
        self.assertIn("no initial board pose", str(caught.exception))

    def test_one_camera_cannot_be_calibrated(self):
        with self.assertRaises(ValueError) as caught:
            self._run(cameras={"camA": self.cameras["camA"]})
        self.assertIn("at least two cameras", str(caught.exception))

    def test_freezing_everything_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._run(optimize_board=False, optimize_extrinsics=False)
        self.assertIn("at least one", str(caught.exception))


class SparsityTests(unittest.TestCase):
    def test_each_observation_block_depends_on_its_camera_and_frame(self):
        board, cameras, _extrinsics, poses, observations = _scene(frames=3)
        prepared = [
            (
                observation.frame,
                observation.camera,
                board[observation.corner_ids],
                observation.pixels,
                cameras[observation.camera].intrinsic_matrix,
            )
            for observation in observations
        ]
        free_cameras = ["camB", "camC"]
        free_frames = sorted(poses)
        sparsity, camera_columns, frame_columns = _build_sparsity(
            prepared, free_cameras, free_frames, "camA", True, True
        )
        self.assertEqual(sparsity.shape[0], 2 * sum(len(item[2]) for item in prepared))
        self.assertEqual(sparsity.shape[1], 6 * (len(free_cameras) + len(free_frames)))
        starts = []
        row = 0
        for item in prepared:
            starts.append(row)
            row += 2 * len(item[2])

        index = next(position for position, item in enumerate(prepared) if item[1] != "camA")
        camera = prepared[index][1]
        frame = prepared[index][0]
        columns = set(sparsity[starts[index]].nonzero()[1])
        expected = (
            set(range(camera_columns[camera], camera_columns[camera] + 6))
            | set(range(frame_columns[frame], frame_columns[frame] + 6))
        )
        self.assertEqual(columns, expected)

        anchor = next(position for position, item in enumerate(prepared) if item[1] == "camA")
        anchor_frame = prepared[anchor][0]
        self.assertEqual(
            set(sparsity[starts[anchor]].nonzero()[1]),
            set(range(frame_columns[anchor_frame], frame_columns[anchor_frame] + 6)),
        )


if __name__ == "__main__":
    unittest.main()
