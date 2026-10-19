"""Calibrated multi-view triangulation for AnimalPoseTracker pose outputs.

The module is deliberately independent of prediction, tracking, and GUI flows.
Callers provide synchronized per-camera poses for the same animal and camera
calibrations. It does not detect cameras, synchronize videos, or assign
cross-camera identities.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


def _readonly_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera calibration with optional Brown-Conrady distortion.

    ``rotation_matrix`` and ``translation_vector`` map world coordinates into
    the camera coordinate frame. ``image_size`` is ``(width, height)`` and is
    metadata; points passed to the projection functions use pixel coordinates
    in the calibration image's coordinate system.
    """

    camera_id: str
    intrinsic_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    distortion_coefficients: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    world_unit: str = "mm"

    def __post_init__(self):
        camera_id = str(self.camera_id).strip()
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        intrinsic = _readonly_array(self.intrinsic_matrix, (3, 3), "intrinsic_matrix")
        rotation = _readonly_array(self.rotation_matrix, (3, 3), "rotation_matrix")
        translation = np.asarray(self.translation_vector, dtype=np.float64).reshape(-1)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation_vector must contain three finite values")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if abs(float(np.linalg.det(intrinsic))) < 1e-12 or abs(float(np.linalg.det(rotation))) < 1e-12:
            raise ValueError("intrinsic and rotation matrices must be invertible")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-2):
            raise ValueError("rotation_matrix must be approximately orthonormal")
        if float(np.linalg.det(rotation)) < 0.0:
            raise ValueError("rotation_matrix must preserve handedness")

        distortion = self.distortion_coefficients
        if distortion is not None:
            distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
            if len(distortion) not in (4, 5, 8) or not np.isfinite(distortion).all():
                raise ValueError("distortion_coefficients must have 4, 5, or 8 finite values")
            object.__setattr__(self, "distortion_coefficients", distortion.copy())

        if self.image_size is not None:
            size = tuple(int(value) for value in self.image_size)
            if len(size) != 2 or min(size) <= 0:
                raise ValueError("image_size must be a positive (width, height) pair")
            object.__setattr__(self, "image_size", size)

        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "intrinsic_matrix", intrinsic)
        object.__setattr__(self, "rotation_matrix", rotation)
        object.__setattr__(self, "translation_vector", translation.copy())
        object.__setattr__(self, "world_unit", str(self.world_unit))

    @property
    def projection_matrix(self) -> np.ndarray:
        """Return the undistorted 3x4 world-to-pixel projection matrix."""
        extrinsic = np.column_stack([self.rotation_matrix, self.translation_vector])
        return self.intrinsic_matrix @ extrinsic

    def to_mapping(self) -> Dict[str, object]:
        """Serialize the calibration values to a plain config-friendly mapping."""
        return {
            "camera_id": self.camera_id,
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.tolist(),
            "distortion_coefficients": (
                None if self.distortion_coefficients is None
                else self.distortion_coefficients.tolist()
            ),
            "image_size": self.image_size,
            "world_unit": self.world_unit,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CameraCalibration":
        """Build a calibration object from a mapping or decoded YAML/JSON block."""
        if not isinstance(value, Mapping):
            raise TypeError("camera calibration must be a mapping")
        return cls(
            camera_id=value["camera_id"],
            intrinsic_matrix=value["intrinsic_matrix"],
            rotation_matrix=value["rotation_matrix"],
            translation_vector=value["translation_vector"],
            distortion_coefficients=value.get("distortion_coefficients"),
            image_size=value.get("image_size"),
            world_unit=value.get("world_unit", "mm"),
        )

    def _distortion(self):
        values = np.zeros((8,), dtype=np.float64)
        if self.distortion_coefficients is not None:
            values[:len(self.distortion_coefficients)] = self.distortion_coefficients
        return values

    def _distort_normalized(self, points: np.ndarray) -> np.ndarray:
        coeff = self._distortion()
        k1, k2, p1, p2, k3, k4, k5, k6 = coeff
        x, y = points[:, 0], points[:, 1]
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = numerator / np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.column_stack([xd, yd])

    def undistort_points(self, points_2d: np.ndarray, iterations: int = 12) -> np.ndarray:
        """Convert pixel points to undistorted normalized camera coordinates."""
        pixels = np.asarray(points_2d, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("points_2d must be a finite (N, 2) pixel array")
        homogeneous = np.column_stack([pixels, np.ones((len(pixels),), dtype=np.float64)])
        normalized_h = homogeneous @ np.linalg.inv(self.intrinsic_matrix).T
        distorted = normalized_h[:, :2] / normalized_h[:, 2:3]
        if self.distortion_coefficients is None:
            return distorted

        coeff = self._distortion()
        k1, k2, p1, p2, k3, k4, k5, k6 = coeff
        estimate = distorted.copy()
        xd, yd = distorted[:, 0], distorted[:, 1]
        for _ in range(max(int(iterations), 1)):
            x, y = estimate[:, 0], estimate[:, 1]
            r2 = x * x + y * y
            r4, r6 = r2 * r2, r2 * r2 * r2
            numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
            radial = numerator / np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
            tangential_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            tangential_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            safe = np.where(np.abs(radial) < 1e-12, 1e-12, radial)
            estimate[:, 0] = (xd - tangential_x) / safe
            estimate[:, 1] = (yd - tangential_y) / safe
        return estimate

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """Project world-space XYZ points to distorted pixel coordinates."""
        points = np.asarray(points_3d, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_3d must be a finite (N, 3) world-coordinate array")
        camera_points = points @ self.rotation_matrix.T + self.translation_vector
        depth = camera_points[:, 2]
        normalized = camera_points[:, :2] / np.where(np.abs(depth[:, None]) < 1e-12, 1e-12, depth[:, None])
        distorted = self._distort_normalized(normalized)
        homogeneous = np.column_stack([distorted, np.ones((len(distorted),), dtype=np.float64)])
        projected = homogeneous @ self.intrinsic_matrix.T
        result = projected[:, :2] / projected[:, 2:3]
        result[depth <= 1e-9] = np.nan
        return result


@dataclass(frozen=True)
class TriangulationResult:
    """Per-keypoint 3D coordinates and reconstruction diagnostics for one frame."""

    points_3d: np.ndarray
    confidence: np.ndarray
    reprojection_error: np.ndarray
    num_views: np.ndarray
    inlier_cameras: Tuple[Tuple[str, ...], ...]

    @property
    def keypoints(self) -> np.ndarray:
        """Return ``(K, 4)`` XYZ + confidence values."""
        return np.column_stack([self.points_3d, self.confidence]).astype(np.float32)


@dataclass(frozen=True)
class TriangulationSequence:
    """Per-frame triangulation results for one already-associated animal track."""

    points_3d: np.ndarray
    confidence: np.ndarray
    reprojection_error: np.ndarray
    num_views: np.ndarray
    inlier_cameras: Tuple[Tuple[Tuple[str, ...], ...], ...]


def _normalize_pose(points: np.ndarray):
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("2D poses must have shape (K, 2) or (K, 3)")
    xy = np.asarray(points[:, :2], dtype=np.float64)
    confidence = (
        np.asarray(points[:, 2], dtype=np.float64)
        if points.shape[1] == 3 else np.ones((len(points),), dtype=np.float64)
    )
    return xy, confidence


def _dlt_triangulate(normalized_points, cameras, weights):
    rows = []
    for point, camera, weight in zip(normalized_points, cameras, weights):
        projection = np.column_stack([camera.rotation_matrix, camera.translation_vector])
        x, y = point
        scale = np.sqrt(max(float(weight), 1e-6))
        rows.append((x * projection[2] - projection[0]) * scale)
        rows.append((y * projection[2] - projection[1]) * scale)
    matrix = np.asarray(rows, dtype=np.float64)
    try:
        _u, _singular, vh = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        return None
    result = homogeneous[:3] / homogeneous[3]
    return result if np.isfinite(result).all() else None


def _reprojection_errors(point_3d, camera_ids, cameras, observed_pixels):
    errors = []
    for camera_id, observed in zip(camera_ids, observed_pixels):
        camera = cameras[camera_id]
        depth = float((camera.rotation_matrix @ point_3d + camera.translation_vector)[2])
        if depth <= 1e-9:
            errors.append(float("inf"))
        else:
            projected = camera.project_points(point_3d[None, :])[0]
            errors.append(float(np.linalg.norm(projected - observed)))
    return np.asarray(errors, dtype=np.float64)


def _refine_point(point, camera_ids, cameras, pixels, scores, max_nfev=40):
    def residual(xyz):
        point_3d = np.asarray(xyz, dtype=np.float64)
        result = []
        for camera_id, observed, score in zip(camera_ids, pixels, scores):
            camera = cameras[camera_id]
            camera_xyz = camera.rotation_matrix @ point_3d + camera.translation_vector
            if camera_xyz[2] <= 1e-9:
                result.extend([1e4, 1e4])
                continue
            projected = camera.project_points(point_3d[None, :])[0]
            result.extend((projected - observed) * np.sqrt(max(float(score), 1e-4)))
        return np.asarray(result, dtype=np.float64)

    result = least_squares(residual, point, loss="soft_l1", f_scale=2.0, max_nfev=max_nfev)
    return result.x if result.success and np.isfinite(result.x).all() else point


@dataclass(frozen=True)
class _PreparedGroup:
    """One associated pose, validated and undistorted once for the solver loop."""

    camera_ids: Tuple[str, ...]
    normalized: Dict[str, np.ndarray]
    pixels: Dict[str, np.ndarray]
    scores: Dict[str, np.ndarray]
    keypoint_count: int


def _prepare_pose_group(poses_by_camera, cameras, *, min_views) -> _PreparedGroup:
    """Validate one group of per-camera poses and undistort them once."""
    if len(poses_by_camera) < int(min_views):
        raise ValueError("poses_by_camera contains fewer cameras than min_views")
    camera_ids = tuple(poses_by_camera.keys())
    missing = [camera_id for camera_id in camera_ids if camera_id not in cameras]
    if missing:
        raise KeyError(f"Missing camera calibration for: {missing}")
    if any(cameras[camera_id].camera_id != camera_id for camera_id in camera_ids):
        raise ValueError("camera mapping keys must match each CameraCalibration.camera_id")
    units = {cameras[camera_id].world_unit for camera_id in camera_ids}
    if len(units) != 1:
        raise ValueError("all camera calibrations must use the same world_unit")
    normalized, pixels, scores = {}, {}, {}
    keypoint_count = None
    for camera_id in camera_ids:
        pose = np.asarray(poses_by_camera[camera_id], dtype=np.float64)
        xy, confidence = _normalize_pose(pose)
        if keypoint_count is None:
            keypoint_count = len(xy)
        elif len(xy) != keypoint_count:
            raise ValueError("all camera poses must contain the same keypoint count")
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(confidence)
        confidence = np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0)
        safe_xy = np.where(valid[:, None], xy, 0.0)
        pixels[camera_id] = safe_xy
        scores[camera_id] = confidence
        normalized[camera_id] = cameras[camera_id].undistort_points(safe_xy)
    return _PreparedGroup(camera_ids, normalized, pixels, scores, int(keypoint_count or 0))


def _available_cameras(group: _PreparedGroup, joint: int, confidence_threshold: float) -> List[str]:
    """Cameras whose observation of ``joint`` is finite and confident enough."""
    return [
        camera_id for camera_id in group.camera_ids
        if group.scores[camera_id][joint] >= confidence_threshold
        and group.scores[camera_id][joint] > 0.0
        and np.isfinite(group.pixels[camera_id][joint]).all()
    ]


def _solve_subset(
    group: _PreparedGroup,
    joint: int,
    camera_ids: Sequence[str],
    cameras: Mapping[str, CameraCalibration],
    backend: str,
) -> Optional[np.ndarray]:
    """Solve one joint from an explicit camera subset; ``None`` when it degenerates."""
    from .accelerate import triangulate_batch

    solved = triangulate_batch(
        np.stack([group.normalized[camera_id][joint] for camera_id in camera_ids])[None],
        np.stack([np.column_stack([cameras[camera_id].rotation_matrix,
                                   cameras[camera_id].translation_vector])
                  for camera_id in camera_ids]),
        np.array([[group.scores[camera_id][joint] for camera_id in camera_ids]]),
        backend=backend,
    )[0]
    return solved if np.isfinite(solved).all() else None


def _drop_outlier_cameras(
    group: _PreparedGroup,
    joint: int,
    point: np.ndarray,
    available: Sequence[str],
    cameras: Mapping[str, CameraCalibration],
    *,
    min_views: int,
    reprojection_threshold: float,
    backend: str,
) -> Tuple[List[str], Optional[np.ndarray]]:
    """Drop the worst reprojecting camera, re-solve, repeat until the rest agree.

    The batched solve has no per-joint RANSAC to pick a subset, so every camera is
    solved first and the odd one out is removed afterwards. Removing one view at a
    time matters: an outlier drags the first solution towards itself, which makes
    the good cameras look bad too, and dropping them all at once would throw the
    joint away. Each pass costs one extra single-point solve and only runs when a
    view actually exceeds the threshold.
    """
    kept = list(available)
    current = point
    while len(kept) > min_views:
        errors = _reprojection_errors(
            current, kept, cameras, [group.pixels[camera_id][joint] for camera_id in kept]
        )
        if np.all(errors <= reprojection_threshold):
            break
        kept.pop(int(np.argmax(errors)))
        current = _solve_subset(group, joint, kept, cameras, backend)
        if current is None:
            return list(available), None
    return kept, current


def triangulate_groups(
    groups: Sequence[Mapping[str, np.ndarray]],
    cameras: Mapping[str, CameraCalibration],
    *,
    confidence_threshold: float = 0.05,
    reprojection_threshold: float = 8.0,
    min_views: int = 2,
    refine: bool = True,
    reject_outliers: bool = True,
    backend: str = "numpy",
) -> List[TriangulationResult]:
    """Triangulate several already-associated poses in one batched pass.

    A frame with N animals is N groups; a recording is every frame's groups laid
    end to end. Every joint of every group that shares a camera set is solved in a
    single call, so one block covers tens of thousands of points — a per-frame call
    would only ever see the handful of joints one animal carries.

    ``backend`` names the solver and is the caller's choice: ``numpy`` needs nothing
    extra, ``numba`` and ``jax-gpu`` need their extras and pay off on large blocks.

    There is no per-joint RANSAC: every available camera is solved first, a view
    that reprojects far off is dropped afterwards (``reject_outliers``), and the
    joint is solved once more from what is left.
    """
    from .accelerate import BACKENDS

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if int(min_views) < 2:
        raise ValueError("min_views must be at least 2")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not np.isfinite(reprojection_threshold) or reprojection_threshold <= 0.0:
        raise ValueError("reprojection_threshold must be finite and positive")
    if not groups:
        return []

    prepared = [_prepare_pose_group(poses, cameras, min_views=min_views) for poses in groups]

    from .accelerate import triangulate_batch

    solved: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    by_camera_set: Dict[Tuple[str, ...], List[Tuple[int, int]]] = {}
    for group_index, group in enumerate(prepared):
        for joint in range(group.keypoint_count):
            available = tuple(_available_cameras(group, joint, confidence_threshold))
            if len(available) >= min_views:
                by_camera_set.setdefault(available, []).append((group_index, joint))
            else:
                solved[(group_index, joint)] = None
    for camera_set, entries in by_camera_set.items():
        points = np.stack([
            np.stack([prepared[group_index].normalized[camera_id][joint]
                      for camera_id in camera_set])
            for group_index, joint in entries
        ])
        projections = np.stack([
            np.column_stack([cameras[camera_id].rotation_matrix,
                             cameras[camera_id].translation_vector])
            for camera_id in camera_set
        ])
        weights = np.stack([
            np.array([prepared[group_index].scores[camera_id][joint]
                      for camera_id in camera_set])
            for group_index, joint in entries
        ])
        rows = triangulate_batch(points, projections, weights, backend=backend)
        for row, entry in enumerate(entries):
            point = rows[row]
            solved[entry] = point if np.isfinite(point).all() else None

    results: List[TriangulationResult] = []
    for group_index, group in enumerate(prepared):
        keypoint_count = group.keypoint_count
        points_3d = np.full((keypoint_count, 3), np.nan, dtype=np.float32)
        confidence_3d = np.zeros((keypoint_count,), dtype=np.float32)
        reprojection = np.full((keypoint_count,), np.nan, dtype=np.float32)
        num_views = np.zeros((keypoint_count,), dtype=np.int32)
        per_joint_inliers = []

        for joint in range(keypoint_count):
            available = _available_cameras(group, joint, confidence_threshold)
            if len(available) < min_views:
                per_joint_inliers.append(tuple())
                continue
            inlier_ids = list(available)
            point = solved.get((group_index, joint))
            if point is not None and reject_outliers:
                inlier_ids, point = _drop_outlier_cameras(
                    group, joint, point, available, cameras,
                    min_views=min_views,
                    reprojection_threshold=reprojection_threshold,
                    backend=backend,
                )
            if point is None:
                per_joint_inliers.append(tuple())
                continue
            if refine:
                point = _refine_point(
                    point,
                    inlier_ids,
                    cameras,
                    [group.pixels[camera_id][joint] for camera_id in inlier_ids],
                    [group.scores[camera_id][joint] for camera_id in inlier_ids],
                )
            errors = _reprojection_errors(
                point, inlier_ids, cameras,
                [group.pixels[camera_id][joint] for camera_id in inlier_ids],
            )
            finite_errors = errors[np.isfinite(errors)]
            if len(finite_errors) < min_views:
                per_joint_inliers.append(tuple())
                continue
            mean_error = float(np.mean(finite_errors))
            points_3d[joint] = point.astype(np.float32)
            num_views[joint] = len(inlier_ids)
            reprojection[joint] = mean_error
            mean_score = float(np.mean([group.scores[camera_id][joint] for camera_id in inlier_ids]))
            confidence_3d[joint] = np.clip(
                mean_score * np.exp(-mean_error / float(reprojection_threshold)), 0.0, 1.0
            )
            per_joint_inliers.append(tuple(inlier_ids))

        results.append(TriangulationResult(
            points_3d=points_3d,
            confidence=confidence_3d,
            reprojection_error=reprojection,
            num_views=num_views,
            inlier_cameras=tuple(per_joint_inliers),
        ))
    return results


def triangulate_keypoints(
    poses_by_camera: Mapping[str, np.ndarray],
    cameras: Mapping[str, CameraCalibration],
    *,
    confidence_threshold: float = 0.05,
    reprojection_threshold: float = 8.0,
    min_views: int = 2,
    refine: bool = True,
    reject_outliers: bool = True,
    backend: str = "numpy",
) -> TriangulationResult:
    """Triangulate one already-associated animal pose from synchronized views.

    Each mapping value is a per-camera ``(K, 2)`` or ``(K, 3)`` pose. The third
    channel, when present, is per-keypoint confidence. All views must refer to
    the same time and same animal identity; cross-camera matching is deliberately
    left to the caller so this geometry module cannot silently combine animals.

    This is :func:`triangulate_groups` with a single group: the joints are solved
    in one batched call, and a camera that reprojects far off is dropped afterwards.
    """
    return triangulate_groups(
        [poses_by_camera], cameras,
        confidence_threshold=confidence_threshold,
        reprojection_threshold=reprojection_threshold,
        min_views=min_views,
        refine=refine,
        reject_outliers=reject_outliers,
        backend=backend,
    )[0]


def triangulate_sequence(
    poses_by_camera: Mapping[str, np.ndarray],
    cameras: Mapping[str, CameraCalibration],
    **kwargs,
) -> TriangulationSequence:
    """Triangulate one already-associated animal sequence across synchronized views.

    Each value must have shape ``(T, K, 2|3)``. Frame indices are assumed to be
    synchronized and keypoint/animal order must match across cameras.
    """
    if not poses_by_camera:
        raise ValueError("poses_by_camera cannot be empty")
    arrays = {camera_id: np.asarray(values) for camera_id, values in poses_by_camera.items()}
    shapes = {camera_id: values.shape[:2] for camera_id, values in arrays.items()}
    if any(values.ndim != 3 or values.shape[-1] not in (2, 3) for values in arrays.values()):
        raise ValueError("each camera sequence must have shape (T, K, 2) or (T, K, 3)")
    if len(set(shapes.values())) != 1:
        raise ValueError("all camera sequences must share frame and keypoint dimensions")
    frame_count, keypoint_count = next(iter(shapes.values()))

    points = np.full((frame_count, keypoint_count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros((frame_count, keypoint_count), dtype=np.float32)
    reprojection = np.full((frame_count, keypoint_count), np.nan, dtype=np.float32)
    num_views = np.zeros((frame_count, keypoint_count), dtype=np.int32)
    # The whole sequence goes in at once so a backend can solve every frame that
    # shares a camera set in one call; without one this is the same per-frame loop.
    groups = [
        {camera_id: values[frame_index] for camera_id, values in arrays.items()}
        for frame_index in range(frame_count)
    ]
    inlier_cameras = []
    for frame_index, frame_result in enumerate(triangulate_groups(groups, cameras, **kwargs)):
        points[frame_index] = frame_result.points_3d
        confidence[frame_index] = frame_result.confidence
        reprojection[frame_index] = frame_result.reprojection_error
        num_views[frame_index] = frame_result.num_views
        inlier_cameras.append(frame_result.inlier_cameras)
    return TriangulationSequence(
        points_3d=points,
        confidence=confidence,
        reprojection_error=reprojection,
        num_views=num_views,
        inlier_cameras=tuple(inlier_cameras),
    )


__all__ = [
    "CameraCalibration",
    "TriangulationResult",
    "TriangulationSequence",
    "triangulate_groups",
    "triangulate_keypoints",
    "triangulate_sequence",
]
