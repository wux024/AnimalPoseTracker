"""Joint refinement of camera extrinsics and per-frame board poses.

This is a small sparse bundle adjustment over a calibration board of known
geometry. Every camera has one pose (world -> camera); every frame has one
board pose (board -> world) shared by the cameras that observed it. Intrinsics
and lens distortion stay fixed: callers pass ideal (undistorted) pixels, the
same grid ``pose3d`` uses after ``CameraRig.undistort``.

The module only imports numpy and scipy. ``pose3d`` sits on the import graph of
``postprocessing.workflows``, which the deployment inference path imports, so
it must keep loading in an environment without the optional ``training`` extra
(torch).

The residual is::

    X_world = R[frame] @ X_board + t[frame]
    X_camera = R[camera] @ X_world + t[camera]
    uv = K[camera] @ X_camera / z
    r = uv[:2] - observed_pixels

with the anchor camera held fixed (the gauge) and a robust loss. The data
contract is explicit and validated on entry: observations are one per
(frame, camera) pair, corner ids are unique and ascending, and a grossly wrong
initial state is rejected before the optimizer runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .core import CameraCalibration

__all__ = ["BundleObservation", "BundleResult", "bundle_adjust"]

Pose = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class BundleObservation:
    """One camera's view of the board in one frame, in ideal (undistorted) pixels."""

    frame: int
    camera: str
    corner_ids: np.ndarray
    pixels: np.ndarray


@dataclass(frozen=True)
class BundleResult:
    """The refined rig and the diagnostics of one optimization run."""

    extrinsics: Dict[str, Pose]
    board_poses: Dict[int, Pose]
    residuals: np.ndarray
    median_px: float
    p90_px: float
    rms_px: float
    cost: float
    nfev: int
    success: bool
    message: str
    parameter_count: int
    #: Per free camera, the six-parameter standard deviation (rotation in rad,
    #: translation in the rig's world unit). Diagnostic only: it assumes the
    #: residuals are independent and the loss is approximately quadratic.
    extrinsic_std: Dict[str, np.ndarray] = field(default_factory=dict)


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix for a rotation vector (axis * angle, radians)."""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_rodrigues` for a proper rotation."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return axis / norm * angle
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
             intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points through a pinhole camera; return pixels and depths."""
    local = np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + translation
    depth = local[:, 2]
    safe = np.where(depth > 1e-9, depth, 1.0)
    pixels = (local @ np.asarray(intrinsic, dtype=np.float64).T)[:, :2] / safe[:, None]
    return pixels, depth


def _pose(value, label: str) -> Pose:
    """Normalise and validate one ``(rotation_matrix, translation)`` pair."""
    try:
        rotation, translation = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a (rotation, translation) pair") from error
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"{label} must hold a 3x3 rotation and a three-value translation")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise ValueError(f"{label} rotation is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
        raise ValueError(f"{label} rotation must have determinant +1")
    return rotation, translation


def _normalise_inputs(
    board_points,
    observations: Sequence[BundleObservation],
    cameras: Mapping[str, CameraCalibration],
    initial_extrinsics: Mapping[str, Pose],
    initial_board_poses: Mapping[int, Pose],
) -> Tuple[np.ndarray, Mapping[str, CameraCalibration], Dict[str, Pose], Dict[int, Pose]]:
    board = np.asarray(board_points, dtype=np.float64)
    if board.ndim != 2 or board.shape[1] != 3 or len(board) < 4 or not np.isfinite(board).all():
        raise ValueError("board_points must be a finite (N, 3) array with at least four points")
    if not isinstance(cameras, Mapping) or len(cameras) < 2:
        raise ValueError("at least two cameras are needed for a joint calibration")
    for name, camera in cameras.items():
        if not isinstance(camera, CameraCalibration) or camera.camera_id != name:
            raise ValueError(f"cameras['{name}'] must be a CameraCalibration whose camera_id matches")

    extrinsics: Dict[str, Pose] = {}
    for name in cameras:
        if name not in initial_extrinsics:
            raise ValueError(f"initial_extrinsics is missing camera '{name}'")
        extrinsics[name] = _pose(initial_extrinsics[name], f"initial extrinsics for '{name}'")
    board_poses = {
        int(frame): _pose(value, f"initial board pose for frame {frame}")
        for frame, value in initial_board_poses.items()
    }
    if not observations:
        raise ValueError("observations must not be empty")
    return board, cameras, extrinsics, board_poses


def _check_observations(
    observations: Sequence[BundleObservation],
    board: np.ndarray,
    cameras: Mapping[str, CameraCalibration],
    board_poses: Mapping[int, Pose],
) -> None:
    seen = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, BundleObservation):
            raise TypeError(f"observation {index} must be a BundleObservation")
        frame = int(observation.frame)
        camera = str(observation.camera)
        if camera not in cameras:
            raise ValueError(f"observation {index} names unknown camera '{camera}'")
        if frame not in board_poses:
            raise ValueError(f"observation {index} has no initial board pose for frame {frame}")
        ids = np.asarray(observation.corner_ids)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError(f"observation {index} corner_ids must be a non-empty integer vector")
        ids = ids.astype(np.int64)
        if int(ids.min()) < 0 or int(ids.max()) >= len(board):
            raise ValueError(
                f"observation {index} corner id outside 0..{len(board) - 1} for frame {frame}"
            )
        if np.any(np.diff(ids) <= 0):
            raise ValueError(f"observation {index} corner_ids must be unique and ascending")
        pixels = np.asarray(observation.pixels, dtype=np.float64)
        if pixels.shape != (len(ids), 2) or not np.isfinite(pixels).all():
            raise ValueError(f"observation {index} pixels must be a finite (N, 2) array")
        if (frame, camera) in seen:
            raise ValueError(f"duplicate observation for frame {frame} and camera '{camera}'")
        seen.add((frame, camera))


def _build_sparsity(prepared, free_cameras, free_frames, anchor: str, optimize_board: bool,
                    optimize_extrinsics: bool):
    """Sparse dependency structure: each residual block touches its camera and frame."""
    rows = 2 * sum(len(item[2]) for item in prepared)
    columns = 6 * (len(free_cameras) + len(free_frames))
    sparsity = lil_matrix((rows, columns), dtype=np.int8)
    camera_columns = {name: 6 * index for index, name in enumerate(free_cameras)}
    frame_columns = {
        frame: 6 * len(free_cameras) + 6 * index for index, frame in enumerate(free_frames)
    }
    row = 0
    for frame, camera, corners, _pixels, _intrinsic in prepared:
        block = slice(row, row + 2 * len(corners))
        if optimize_extrinsics and camera != anchor:
            start = camera_columns[camera]
            sparsity[block, start:start + 6] = 1
        if optimize_board:
            start = frame_columns[frame]
            sparsity[block, start:start + 6] = 1
        row += 2 * len(corners)
    return sparsity, camera_columns, frame_columns


def bundle_adjust(
    board_points: np.ndarray,
    observations: Sequence[BundleObservation],
    cameras: Mapping[str, CameraCalibration],
    initial_extrinsics: Mapping[str, Pose],
    initial_board_poses: Mapping[int, Pose],
    *,
    anchor: str,
    optimize_board: bool = True,
    optimize_extrinsics: bool = True,
    loss: str = "cauchy",
    f_scale: float = 5.0,
    max_nfev: int = 200,
    max_initial_residual_px: float = 10.0,
) -> BundleResult:
    """Refine extrinsics and board poses together from ideal-pixel observations.

    ``anchor`` is held fixed: it fixes the world frame and removes the six-degree
    gauge freedom. Every other camera is free (six parameters each) when
    ``optimize_extrinsics`` is set; every observed frame is free (six parameters
    each) when ``optimize_board`` is set. The two switches also drive the staged
    runs (board only, then extrinsics only, then both).

    The initial state is rejected when its residual median exceeds
    ``max_initial_residual_px``: a data-assembly mistake shows up as a huge
    initial residual, and continuing from it would only fit the mistake.
    """
    board, cameras, extrinsics, board_poses = _normalise_inputs(
        board_points, observations, cameras, initial_extrinsics, initial_board_poses
    )
    if anchor not in cameras:
        raise ValueError(f"anchor camera '{anchor}' is not in cameras")
    if not optimize_board and not optimize_extrinsics:
        raise ValueError("at least one of optimize_board and optimize_extrinsics must be set")
    _check_observations(observations, board, cameras, board_poses)

    prepared = [
        (
            int(observation.frame),
            str(observation.camera),
            board[np.asarray(observation.corner_ids, dtype=np.int64)],
            np.asarray(observation.pixels, dtype=np.float64),
            cameras[str(observation.camera)].intrinsic_matrix,
        )
        for observation in observations
    ]
    free_cameras = [name for name in sorted(cameras) if name != anchor] if optimize_extrinsics else []
    free_frames = sorted({item[0] for item in prepared}) if optimize_board else []
    if optimize_extrinsics and not free_cameras:
        raise ValueError("optimize_extrinsics needs at least one camera besides the anchor")
    if optimize_board and not free_frames:
        raise ValueError("optimize_board needs at least one observed frame")

    blocks = []
    if optimize_extrinsics:
        for name in free_cameras:
            rotation, translation = extrinsics[name]
            blocks.append(np.concatenate([_rotation_vector(rotation), translation]))
    if optimize_board:
        for frame in free_frames:
            rotation, translation = board_poses[frame]
            blocks.append(np.concatenate([_rotation_vector(rotation), translation]))
    theta = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    def unpack(params: np.ndarray):
        current_extrinsics = dict(extrinsics)
        current_poses = dict(board_poses)
        offset = 0
        if optimize_extrinsics:
            for name in free_cameras:
                rotation = _rodrigues(params[offset:offset + 3])
                translation = params[offset + 3:offset + 6]
                offset += 6
                current_extrinsics[name] = (rotation, translation)
        if optimize_board:
            for frame in free_frames:
                rotation = _rodrigues(params[offset:offset + 3])
                translation = params[offset + 3:offset + 6]
                offset += 6
                current_poses[frame] = (rotation, translation)
        return current_extrinsics, current_poses

    def residual(params: np.ndarray) -> np.ndarray:
        current_extrinsics, current_poses = unpack(params)
        parts = []
        for frame, camera, corners, pixels, intrinsic in prepared:
            board_rotation, board_translation = current_poses[frame]
            camera_rotation, camera_translation = current_extrinsics[camera]
            world = corners @ board_rotation.T + board_translation
            projected, depth = _project(world, camera_rotation, camera_translation, intrinsic)
            difference = projected - pixels
            behind = depth <= 1e-9
            if behind.any():
                difference[behind] = 1e4
            parts.append(difference.reshape(-1))
        return np.concatenate(parts)

    initial = residual(theta).reshape(-1, 2)
    norms = np.linalg.norm(initial, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("the initial residual is not finite")
    median = float(np.median(norms))
    if median > float(max_initial_residual_px):
        per_camera: Dict[str, list] = {}
        row = 0
        for _frame, camera, corners, _pixels, _intrinsic in prepared:
            per_camera.setdefault(camera, []).append(norms[row:row + len(corners)])
            row += len(corners)
        detail = ", ".join(
            f"{name}: {float(np.median(np.concatenate(values))):.1f} px"
            for name, values in per_camera.items()
        )
        raise ValueError(
            f"initial residual median is {median:.1f} px (limit {max_initial_residual_px:g} px); "
            f"check the board poses and the frame/corner mapping ({detail})"
        )

    sparsity, camera_columns, frame_columns = _build_sparsity(
        prepared, free_cameras, free_frames, anchor, optimize_board, optimize_extrinsics
    )
    result = least_squares(
        residual,
        theta,
        jac_sparsity=sparsity,
        method="trf",
        tr_solver="lsmr",
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
        max_nfev=max_nfev,
    )

    refined_extrinsics, refined_poses = unpack(result.x)
    raw = residual(result.x).reshape(-1, 2)
    distances = np.linalg.norm(raw, axis=1)
    standard_deviations: Dict[str, np.ndarray] = {}
    if optimize_extrinsics and result.jac is not None:
        try:
            normal = result.jac.T @ result.jac
            if hasattr(normal, "toarray"):
                normal = normal.toarray()
            degrees = max(len(result.fun) - theta.size, 1)
            variance = float(np.sum(np.asarray(result.fun) ** 2) / degrees)
            covariance = np.linalg.pinv(np.asarray(normal, dtype=np.float64)) * variance
            errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
            for name, column in camera_columns.items():
                standard_deviations[name] = errors[column:column + 6].copy()
        except (np.linalg.LinAlgError, ValueError):
            standard_deviations = {}

    return BundleResult(
        extrinsics=refined_extrinsics,
        board_poses=refined_poses,
        residuals=raw,
        median_px=float(np.median(distances)),
        p90_px=float(np.percentile(distances, 90)),
        rms_px=float(np.sqrt(np.mean(raw ** 2))),
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        parameter_count=int(theta.size),
        extrinsic_std=standard_deviations,
    )
