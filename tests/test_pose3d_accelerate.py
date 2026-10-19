"""The batched triangulation must agree with the per-point one, to the last digit."""

import numpy as np
import pytest

from animalposetracker.pose3d.accelerate import (
    available_backends,
    jax_available,
    numba_available,
    triangulate_batch,
)
from animalposetracker.pose3d.core import CameraCalibration, _dlt_triangulate


def _camera(name, translation, focal=800.0, center=(320.0, 240.0)):
    intrinsic = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        camera_id=name,
        intrinsic_matrix=intrinsic,
        rotation_matrix=np.eye(3),
        translation_vector=np.asarray(translation, dtype=np.float64),
    )


def _rig():
    return [
        _camera("a", [0.0, 0.0, 0.0]),
        _camera("b", [-80.0, 0.0, 0.0]),
        _camera("c", [0.0, 60.0, 0.0]),
    ]


def _scene(count=7, seed=0):
    cameras = _rig()
    rng = np.random.default_rng(seed)
    points_3d = rng.uniform(-50.0, 50.0, size=(count, 3)) + np.array([0.0, 0.0, 500.0])
    # `_dlt_triangulate` takes normalised image coordinates, not pixels: the projection
    # matrices it builds are [R | t], with the intrinsics already divided out upstream.
    observations = []
    for camera in cameras:
        rotated = camera.rotation_matrix @ points_3d.T + camera.translation_vector[:, None]
        observations.append((rotated[:2] / rotated[2]).T)
    observations = np.stack(observations, axis=1)
    projection = np.stack(
        [np.column_stack([camera.rotation_matrix, camera.translation_vector]) for camera in cameras]
    )
    return cameras, points_3d, observations, projection


def test_batch_matches_the_per_point_implementation():
    cameras, points_3d, observations, projection = _scene()
    weights = np.ones(observations.shape[:2])

    expected = np.array([
        _dlt_triangulate(observations[index], cameras, weights[index])
        for index in range(len(points_3d))
    ])
    got = triangulate_batch(observations, projection, weights, backend="numpy")

    assert np.allclose(got, expected, atol=1e-9)
    assert np.allclose(got, points_3d, atol=1e-6)


def test_a_degenerate_row_comes_back_as_nan():
    _, _, observations, projection = _scene()
    observations = observations.copy()
    observations[3] = 0.0

    got = triangulate_batch(observations, projection, backend="numpy")

    assert np.isnan(got[3]).all()
    assert np.isfinite(got[:3]).all()


def test_backend_validation():
    _, _, observations, projection = _scene()

    with pytest.raises(ValueError):
        triangulate_batch(observations, projection, backend="cuda")

    if not jax_available():
        with pytest.raises(RuntimeError):
            triangulate_batch(observations, projection, backend="jax-gpu")
    if not numba_available():
        with pytest.raises(RuntimeError):
            triangulate_batch(observations, projection, backend="numba")


def test_available_backends_reports_numpy_always():
    reported = available_backends()

    assert reported["numpy"] is True
    assert set(reported) == {"numpy", "numba", "jax-gpu"}


@pytest.mark.skipif(not jax_available(), reason="needs a jax GPU build and a device")
def test_jax_gpu_matches_numpy():
    _, _, observations, projection = _scene()

    baseline = triangulate_batch(observations, projection, backend="numpy")
    accelerated = triangulate_batch(observations, projection, backend="jax-gpu")

    assert np.allclose(accelerated, baseline, atol=1e-9)


@pytest.mark.skipif(not numba_available(), reason="needs the pose3d-numba extra")
def test_numba_matches_numpy():
    _, _, observations, projection = _scene()

    baseline = triangulate_batch(observations, projection, backend="numpy")
    accelerated = triangulate_batch(observations, projection, backend="numba")

    assert np.allclose(accelerated, baseline, atol=1e-9)


def _pixel_poses(observations, count):
    """Normalised observations to ``(K, 3)`` pixel poses for ``triangulate_keypoints``."""
    pixels = observations * np.array([800.0, 800.0]) + np.array([320.0, 240.0])
    return {
        name: np.column_stack([pixels[:, index], np.ones(count)])
        for index, name in enumerate(("a", "b", "c"))
    }


def test_the_batched_path_recovers_the_scene():
    from animalposetracker.pose3d.core import triangulate_keypoints

    cameras, points_3d, observations, _ = _scene(count=9)
    mapping = {camera.camera_id: camera for camera in cameras}
    poses = _pixel_poses(observations, len(points_3d))

    result = triangulate_keypoints(poses, mapping, refine=False, reject_outliers=False)

    assert np.allclose(result.points_3d, points_3d, atol=1e-3)
    assert result.num_views.tolist() == [3] * len(points_3d)


def test_joints_with_different_camera_sets_are_grouped():
    from animalposetracker.pose3d.core import triangulate_keypoints

    cameras, points_3d, observations, _ = _scene(count=9)
    mapping = {camera.camera_id: camera for camera in cameras}
    poses = _pixel_poses(observations, len(points_3d))
    poses["c"][:4, 2] = 0.0  # the first four joints are seen by two cameras only

    result = triangulate_keypoints(poses, mapping, refine=False)

    assert result.num_views[:4].tolist() == [2, 2, 2, 2]
    assert result.num_views[4:].tolist() == [3] * 5
    assert np.allclose(result.points_3d, points_3d, atol=1e-3)


def test_a_bad_camera_is_dropped_after_the_batch():
    from animalposetracker.pose3d.core import triangulate_keypoints

    cameras, points_3d, observations, _ = _scene(count=5)
    mapping = {camera.camera_id: camera for camera in cameras}
    poses = _pixel_poses(observations, len(points_3d))
    poses["c"][0, 0] += 120.0  # the first joint is 120 px off in camera c

    result = triangulate_keypoints(poses, mapping, reprojection_threshold=5.0)

    assert result.num_views[0] == 2
    assert "c" not in result.inlier_cameras[0]
    assert np.allclose(result.points_3d[0], points_3d[0], atol=1e-2)
    assert result.num_views[1:].tolist() == [3] * 4


def test_every_group_of_a_block_is_solved_and_kept_apart():
    from animalposetracker.pose3d.core import triangulate_groups

    cameras, points_3d, observations, _ = _scene(count=6)
    mapping = {camera.camera_id: camera for camera in cameras}
    first = _pixel_poses(observations, len(points_3d))
    second = {name: np.column_stack([pose[:, 0] + 40.0, pose[:, 1], pose[:, 2]])
              for name, pose in first.items()}

    solved = triangulate_groups([first, second], mapping, refine=False)

    assert len(solved) == 2
    for result in solved:
        assert result.num_views.tolist() == [3] * len(points_3d)
    assert not np.allclose(solved[0].points_3d, solved[1].points_3d)
