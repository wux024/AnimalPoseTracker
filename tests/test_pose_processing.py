import unittest

import numpy as np

from animalposetracker.postprocessing.filters import filter_pose_2d, filter_pose_3d
from animalposetracker.pose3d import (
    CameraCalibration,
    triangulate_keypoints,
    triangulate_sequence,
)


def camera(camera_id, translation, distortion=None):
    return CameraCalibration(
        camera_id=camera_id,
        intrinsic_matrix=np.asarray(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        rotation_matrix=np.eye(3),
        translation_vector=np.asarray(translation, dtype=np.float64),
        distortion_coefficients=distortion,
        image_size=(640, 480),
        world_unit="mm",
    )


class PoseFilterTests(unittest.TestCase):
    def test_median_filter_removes_a_spike_without_mutating_raw_predictions(self):
        poses = np.zeros((9, 2, 3), dtype=np.float32)
        poses[:, :, 0] = np.arange(9, dtype=np.float32)[:, None]
        poses[:, :, 1] = 2.0 * np.arange(9, dtype=np.float32)[:, None]
        poses[:, :, 2] = 0.9
        poses[4, 0, :2] = [100.0, -100.0]
        raw = poses.copy()

        filtered = filter_pose_2d(poses, method="median", window_length=5)

        self.assertEqual(filtered.shape, poses.shape)
        self.assertLess(abs(float(filtered[4, 0, 0]) - 4.0), 3.0)
        np.testing.assert_array_equal(poses, raw)

    def test_confidence_mask_and_short_gap_interpolation(self):
        poses = np.zeros((5, 1, 3), dtype=np.float32)
        poses[:, 0, 0] = np.arange(5, dtype=np.float32)
        poses[:, 0, 1] = 2.0 * np.arange(5, dtype=np.float32)
        poses[:, 0, 2] = 0.9
        poses[2, 0] = [100.0, 100.0, 0.0]

        filtered = filter_pose_2d(
            poses,
            method="none",
            confidence_threshold=0.25,
            max_gap=1,
        )

        np.testing.assert_allclose(filtered[2, 0, :2], [2.0, 4.0])
        self.assertEqual(float(filtered[2, 0, 2]), 0.0)

    def test_3d_filter_accepts_tracks_axis_and_preserves_confidence(self):
        poses = np.zeros((7, 3, 4), dtype=np.float32)
        poses[:, :, 0] = np.arange(7, dtype=np.float32)[:, None]
        poses[:, :, 1] = 1.0
        poses[:, :, 2] = 2.0
        poses[:, :, 3] = 0.8
        poses[3, 1, 0] = 50.0

        filtered = filter_pose_3d(poses, method="median", window_length=3)

        self.assertEqual(filtered.shape, poses.shape)
        self.assertLess(float(filtered[3, 1, 0]), 10.0)
        np.testing.assert_array_equal(filtered[..., 3], poses[..., 3])

    def test_filter_rejects_invalid_confidence_and_window_options(self):
        poses = np.zeros((4, 1, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "confidence channel"):
            filter_pose_2d(poses, confidence_threshold=0.2)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            filter_pose_2d(poses, window_length=1)


class Pose3DTests(unittest.TestCase):
    def setUp(self):
        self.cameras = {
            "left": camera("left", [0.0, 0.0, 0.0]),
            "right": camera("right", [-1.0, 0.0, 0.0]),
            "top": camera("top", [0.0, -1.0, 0.0]),
        }

    def test_multiview_triangulation_recovers_world_coordinates(self):
        expected = np.asarray([[0.2, 0.15, 5.0], [-0.3, 0.1, 4.0]], dtype=np.float64)
        poses = {
            camera_id: np.column_stack([
                calibration.project_points(expected),
                np.asarray([0.95, 0.8]),
            ])
            for camera_id, calibration in self.cameras.items()
        }

        result = triangulate_keypoints(poses, self.cameras, refine=True)

        np.testing.assert_allclose(result.points_3d, expected, atol=1e-5)
        self.assertTrue(np.all(result.num_views == 3))
        self.assertTrue(np.all(result.reprojection_error < 1e-5))
        self.assertEqual(result.inlier_cameras[0], ("left", "right", "top"))
        self.assertTrue(np.all(result.confidence > 0.0))

    def test_ransac_drops_a_bad_camera_observation(self):
        expected = np.asarray([[0.2, 0.15, 5.0]], dtype=np.float64)
        cameras = dict(self.cameras)
        cameras["rear"] = camera("rear", [1.0, 1.0, 0.0])
        poses = {}
        for camera_id, calibration in cameras.items():
            pixel = calibration.project_points(expected)[0]
            if camera_id == "rear":
                pixel = pixel + np.asarray([150.0, -100.0])
            poses[camera_id] = np.asarray([[pixel[0], pixel[1], 0.9]], dtype=np.float64)

        result = triangulate_keypoints(
            poses, cameras, reprojection_threshold=2.0, ransac=True, refine=True
        )

        np.testing.assert_allclose(result.points_3d[0], expected[0], atol=1e-4)
        self.assertEqual(result.num_views[0], 3)
        self.assertNotIn("rear", result.inlier_cameras[0])

    def test_distortion_round_trip_and_calibration_mapping(self):
        distorted_camera = camera("cam", [0.0, 0.0, 0.0], [0.08, -0.02, 0.001, -0.002, 0.003])
        point = np.asarray([[0.3, -0.2, 3.0]], dtype=np.float64)
        pixel = distorted_camera.project_points(point)
        normalized = distorted_camera.undistort_points(pixel)
        np.testing.assert_allclose(normalized, point[:, :2] / point[:, 2:3], atol=1e-8)

        restored = CameraCalibration.from_mapping(distorted_camera.to_mapping())
        np.testing.assert_allclose(restored.project_points(point), pixel, atol=1e-8)

    def test_insufficient_views_remain_missing_and_sequence_shape_is_preserved(self):
        one_view = {
            "left": np.asarray([[320.0, 240.0, 0.9]], dtype=np.float64),
        }
        with self.assertRaisesRegex(ValueError, "fewer cameras than min_views"):
            triangulate_keypoints(one_view, self.cameras)
        missing_joint = {
            "left": np.asarray([[320.0, 240.0, 0.9]], dtype=np.float64),
            "right": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            "top": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        }
        result = triangulate_keypoints(missing_joint, self.cameras)
        self.assertTrue(np.isnan(result.points_3d[0]).all())
        self.assertEqual(result.num_views[0], 0)

        world = np.asarray([[0.2, 0.1, 5.0]], dtype=np.float64)
        sequences = {
            camera_id: np.stack([
                np.column_stack([calibration.project_points(world), [0.9]]),
                np.column_stack([calibration.project_points(world + [0.1, 0.0, 0.0]), [0.8]]),
            ])
            for camera_id, calibration in self.cameras.items()
        }
        result = triangulate_sequence(sequences, self.cameras)
        self.assertEqual(result.points_3d.shape, (2, 1, 3))
        np.testing.assert_allclose(result.points_3d[:, 0], [[0.2, 0.1, 5.0], [0.3, 0.1, 5.0]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
