"""Calibrate a multi-camera rig from ChArUco frames or videos.

The command hides the whole calibration path behind one entry point:

1. detect the ChArUco corners in every camera's frames (a folder of images or a
   video, sampled every ``--frame-step`` frames);
2. fit one constrained intrinsics model per camera (fixed principal point,
   ``fx == fy``, no tangential terms) unless an existing ``cameras.yaml`` is
   reused with ``--intrinsics-from``;
3. initialise the extrinsics from frame-paired PnP (world = anchor camera) and
   the board poses from the anchor camera;
4. refine one extrinsic per camera plus one board pose per frame with the sparse
   bundle adjustment in :mod:`animalposetracker.pose3d.bundle`;
5. validate on a temporal holdout, refit on every usable frame, and write a
   ``cameras.yaml`` that :func:`load_camera_rig` reads back.

The command is deliberately conservative: it never overwrites an existing
calibration without ``--force`` and it fails with a named camera/frame when the
input cannot support a calibration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import yaml

from .bundle import BundleObservation, bundle_adjust
from .calibration import CalibrationError, CameraRig, load_camera_rig
from .core import CameraCalibration, triangulate_keypoints

__all__ = [
    "BoardSpec",
    "IntrinsicFit",
    "CalibrationResult",
    "detect_views",
    "fit_intrinsics",
    "calibrate_detections",
    "write_camera_rig",
    "run",
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_DICTIONARIES = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}
_CANDIDATES = [
    ("none", 0),
    ("brown", 1),
    ("brown", 2),
    ("brown", 5),
    ("fisheye", 1),
    ("fisheye", 2),
    ("fisheye", 4),
]


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board: squares, physical sizes, and the ArUco dictionary."""

    squares_x: int = 10
    squares_y: int = 7
    square_mm: float = 22.0
    marker_mm: float = 16.5
    dictionary: str = "DICT_4X4_50"

    def __post_init__(self):
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("a ChArUco board needs at least two squares per side")
        if self.square_mm <= 0.0 or self.marker_mm <= 0.0:
            raise ValueError("square_mm and marker_mm must be positive")
        if self.marker_mm >= self.square_mm:
            raise ValueError("marker_mm must be smaller than square_mm")
        if self.dictionary.upper() not in _DICTIONARIES:
            raise ValueError(f"unknown ArUco dictionary '{self.dictionary}'")

    def board(self) -> "cv2.aruco.CharucoBoard":
        dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARIES[self.dictionary.upper()])
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), float(self.square_mm),
            float(self.marker_mm), dictionary,
        )

    def points(self) -> np.ndarray:
        """Corner coordinates in millimetres, indexed by ChArUco corner id."""
        return self.board().getChessboardCorners().astype(np.float64)


@dataclass(frozen=True)
class IntrinsicFit:
    """One camera's pinhole matrix and lens model."""

    matrix: np.ndarray
    model: str
    coefficients: np.ndarray
    rms: float = float("nan")


@dataclass(frozen=True)
class CalibrationResult:
    """The refined rig, the lens fits, and the validation metrics."""

    cameras: Dict[str, CameraCalibration]
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]]
    intrinsics: Dict[str, IntrinsicFit]
    world_unit: str
    sync_mode: str
    metrics: Dict[str, object]

    def rig(self) -> CameraRig:
        return CameraRig(
            cameras=dict(self.cameras),
            distortion={name: (fit.model, fit.coefficients) for name, fit in self.intrinsics.items()},
            world_unit=self.world_unit,
            sync_mode=self.sync_mode,
        )


@dataclass(frozen=True)
class View:
    """One camera's input path (image folder or video file)."""

    camera: str
    path: Path


def _iter_frames(path: Path, frame_step: int) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield ``(frame index, grayscale image)`` from an image folder or a video."""
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"no images in '{path}'")
        for index, file in enumerate(files):
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                yield index, image
        return
    if path.is_file():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video '{path}'")
        step = max(1, int(frame_step or 30))
        index = -1
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if index % step:
                    continue
                yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        finally:
            capture.release()
        return
    raise FileNotFoundError(f"calibration input does not exist: {path}")


def _detect(
    gray: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
    detector: "cv2.aruco.ArucoDetector",
    min_corners: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) < 5:
        return None
    count, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    if not count or count < min_corners:
        return None
    return ids.ravel().astype(np.int64), corners.reshape(-1, 2).astype(np.float64)


def detect_views(
    views: Sequence[View],
    spec: BoardSpec,
    *,
    frame_step: int = 0,
    max_frames: int = 300,
    min_corners: int = 12,
) -> Tuple[Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]], Dict[str, Tuple[int, int]]]:
    """Detect the board in every view; return observations and image sizes."""
    if len(views) < 2:
        raise ValueError("at least two cameras are needed for a calibration")
    board = spec.board()
    dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARIES[spec.dictionary.upper()])
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    detections: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    sizes: Dict[str, Tuple[int, int]] = {}
    for view in views:
        entries: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        size = None
        processed = 0
        for frame, gray in tqdm(
                _iter_frames(view.path, frame_step),
                desc=f"detect {view.camera}", unit="frame", disable=None,
                dynamic_ncols=True, leave=False):
            if max_frames and processed >= max_frames:
                break
            processed += 1
            if size is None:
                size = (int(gray.shape[1]), int(gray.shape[0]))
            detection = _detect(gray, board, detector, min_corners)
            if detection is not None:
                entries[frame] = detection
        if not entries:
            raise ValueError(f"camera '{view.camera}' has no frame with at least {min_corners} corners")
        detections[view.camera] = entries
        sizes[view.camera] = size
    common = sorted(set.intersection(*(set(entries) for entries in detections.values())))
    if len(common) < 4:
        counts = {name: len(entries) for name, entries in detections.items()}
        raise ValueError(
            f"only {len(common)} frames are seen by every camera (counts: {counts}); "
            "at least four common frames are needed"
        )
    return detections, sizes


def _fit_one(objects, images, size: Tuple[int, int], model: str, coefficients: int):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    if model == "fisheye":
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        for level, flag in ((2, cv2.fisheye.CALIB_FIX_K2), (3, cv2.fisheye.CALIB_FIX_K3),
                            (4, cv2.fisheye.CALIB_FIX_K4)):
            if coefficients < level:
                flags |= flag
        rms, matrix, distortion, _, _ = cv2.fisheye.calibrate(
            [item.reshape(-1, 1, 3).astype(np.float64) for item in objects],
            [item.reshape(-1, 1, 2).astype(np.float64) for item in images],
            (int(size[0]), int(size[1])), None, None, flags=flags, criteria=criteria,
        )
        return matrix, np.asarray(distortion).reshape(-1)[:4], float(rms)
    if model not in {"none", "brown"}:
        raise ValueError(f"unknown intrinsic model '{model}'")
    flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO
    if coefficients < 2:
        flags |= cv2.CALIB_FIX_K2
    if coefficients < 3:
        flags |= cv2.CALIB_FIX_K3
    if coefficients == 0:
        flags |= cv2.CALIB_FIX_K1
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        [item.reshape(-1, 1, 3).astype(np.float32) for item in objects],
        [item.reshape(-1, 1, 2).astype(np.float32) for item in images],
        (int(size[0]), int(size[1])), None, None, flags=flags, criteria=criteria,
    )
    values = np.zeros(5) if coefficients == 0 else np.asarray(distortion).reshape(-1)[:5]
    return matrix, values, float(rms)


def _fit_candidates(objects, images, size, candidates):
    results = []
    for name, count in candidates:
        try:
            matrix, distortion, rms = _fit_one(objects, images, size, name, count)
        except cv2.error:
            continue
        results.append((rms, name, count, matrix, distortion))
    return results


def _choose_intrinsic(results) -> IntrinsicFit:
    """Lowest RMS; among fits within 1% prefer fewer distortion coefficients."""
    best = min(item[0] for item in results)
    near = [item for item in results if item[0] <= best * 1.01]
    rms, name, count, matrix, distortion = min(near, key=lambda item: (item[2], item[1] != "none"))
    return IntrinsicFit(matrix, name, np.asarray(distortion, dtype=np.float64), float(rms))


def fit_intrinsics(
    entries: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    size: Tuple[int, int],
    spec: BoardSpec,
    model: str = "auto",
    max_frames: int = 60,
) -> IntrinsicFit:
    """Fit one constrained lens model on at most ``max_frames`` evenly spaced frames.

    OpenCV's ``calibrateCamera`` becomes disproportionately slower with many
    frames (about 3 s for 60 frames and 145 s for 200 on the reference data), so
    the fit uses a spread of at most 60 frames. The extrinsics still use every
    detected frame.
    """
    items = list(entries.items())
    if max_frames and len(items) > int(max_frames):
        picks = np.linspace(0, len(items) - 1, int(max_frames)).round().astype(int)
        items = [items[index] for index in picks]
    points = spec.points()
    objects = [points[ids] for _frame, (ids, _pixels) in items]
    images = [pixels for _frame, (_ids, pixels) in items]
    if len(objects) < 3:
        raise ValueError(
            f"only {len(objects)} frames with corners; at least three are needed to fit intrinsics"
        )
    if model == "auto":
        results = _fit_candidates(objects, images, size, [("brown", 1), ("fisheye", 1)])
        if not results or min(item[0] for item in results) > 0.5:
            results.extend(_fit_candidates(
                objects, images, size, [("none", 0), ("brown", 5), ("fisheye", 4)]
            ))
    else:
        if model not in {"none", "brown", "fisheye"}:
            raise ValueError(f"intrinsic model must be auto, none, brown or fisheye; got '{model}'")
        results = _fit_candidates(
            objects, images, size, [(name, count) for name, count in _CANDIDATES if name == model]
        )
    if not results:
        raise ValueError("no lens model could be fitted to the detected corners")
    return _choose_intrinsic(results)


def _undistort(fit: IntrinsicFit, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    if fit.model == "none" or not np.any(fit.coefficients):
        return pixels.copy()
    points = np.ascontiguousarray(pixels.reshape(-1, 1, 2))
    if fit.model == "fisheye":
        fixed = cv2.fisheye.undistortPoints(points, fit.matrix, fit.coefficients.reshape(1, -1), P=fit.matrix)
    else:
        fixed = cv2.undistortPoints(points, fit.matrix, fit.coefficients.reshape(1, -1), P=fit.matrix)
    return fixed.reshape(-1, 2)


def _pnp(ideal: np.ndarray, object_points: np.ndarray, matrix: np.ndarray):
    ok, rvec, tvec = cv2.solvePnP(
        np.ascontiguousarray(object_points.reshape(-1, 1, 3)),
        np.ascontiguousarray(ideal.reshape(-1, 1, 2)),
        np.ascontiguousarray(matrix),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3)


def _frame_poses(
    detections: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    intrinsics: Mapping[str, IntrinsicFit],
    points: np.ndarray,
    frames: Sequence[int],
    camera: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for frame in frames:
        entry = detections[camera].get(frame)
        if entry is None:
            continue
        ids, pixels = entry
        ideal = _undistort(intrinsics[camera], pixels)
        pose = _pnp(ideal, points[ids], intrinsics[camera].matrix)
        if pose is not None:
            poses[frame] = pose
    return poses


def initial_extrinsics(
    detections: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    intrinsics: Mapping[str, IntrinsicFit],
    points: np.ndarray,
    frames: Sequence[int],
    anchor: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Frame-paired PnP median extrinsics; the anchor is identity by definition."""
    per_camera = {name: _frame_poses(detections, intrinsics, points, frames, name) for name in detections}
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {anchor: (np.eye(3), np.zeros(3))}
    for name in detections:
        if name == anchor:
            continue
        rotations, translations = [], []
        for frame in frames:
            if frame not in per_camera[anchor] or frame not in per_camera[name]:
                continue
            anchor_rotation, anchor_translation = per_camera[anchor][frame]
            rotation_camera, translation_camera = per_camera[name][frame]
            rotation = rotation_camera @ anchor_rotation.T
            rotations.append(rotation)
            translations.append(translation_camera - rotation @ anchor_translation)
        if not rotations:
            raise ValueError(f"camera '{name}' shares no usable frame with the anchor '{anchor}'")
        mean = np.mean(rotations, axis=0)
        u, _, vt = np.linalg.svd(mean)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        extrinsics[name] = (rotation, np.median(np.asarray(translations), axis=0))
    return extrinsics


def _board_poses(
    detections: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    intrinsics: Mapping[str, IntrinsicFit],
    points: np.ndarray,
    frames: Sequence[int],
    extrinsics: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    anchor: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    anchor_rotation, anchor_translation = extrinsics[anchor]
    poses = {}
    for frame in frames:
        entry = detections[anchor].get(frame)
        if entry is None:
            continue
        ids, pixels = entry
        pose = _pnp(_undistort(intrinsics[anchor], pixels), points[ids], intrinsics[anchor].matrix)
        if pose is None:
            continue
        rotation, translation = pose
        poses[frame] = (
            anchor_rotation.T @ rotation,
            anchor_rotation.T @ (translation - anchor_translation),
        )
    return poses


def _observations(
    detections: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    intrinsics: Mapping[str, IntrinsicFit],
    frames: Sequence[int],
) -> List[BundleObservation]:
    observations = []
    for frame in frames:
        for camera in detections:
            entry = detections[camera].get(frame)
            if entry is None:
                continue
            ids, pixels = entry
            order = np.argsort(ids)
            observations.append(
                BundleObservation(frame, camera, ids[order].astype(np.int64), _undistort(intrinsics[camera], pixels[order]))
            )
    return observations


def _rig_with(
    extrinsics: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    intrinsics: Mapping[str, IntrinsicFit],
    image_sizes: Mapping[str, Tuple[int, int]],
    world_unit: str,
    sync_mode: str,
) -> CameraRig:
    """Build a rig from per-camera intrinsics and extrinsics."""
    return CameraRig(
        cameras={
            name: CameraCalibration(
                name, intrinsics[name].matrix, rotation, translation,
                image_size=image_sizes.get(name), world_unit=world_unit,
            )
            for name, (rotation, translation) in extrinsics.items()
        },
        distortion={name: (fit.model, fit.coefficients) for name, fit in intrinsics.items()},
        world_unit=world_unit,
        sync_mode=sync_mode,
    )


def board_metrics(rig, detections, intrinsics, frames, board_points):
    """Triangulate the board on ``frames`` and compare pairwise distances with the model."""
    errors, reprojections, used = [], [], 0
    for frame in frames:
        ids = None
        for camera in detections:
            entry = detections[camera].get(frame)
            if entry is None:
                ids = None
                break
            current = set(entry[0].tolist())
            ids = current if ids is None else ids & current
        if not ids or len(ids) < 6:
            continue
        ordered = np.asarray(sorted(ids), dtype=np.int64)
        poses = {}
        for camera in detections:
            entry = detections[camera][frame]
            order = np.argsort(entry[0])
            positions = np.searchsorted(entry[0][order], ordered)
            pixels = entry[1][order][positions]
            poses[camera] = np.column_stack([_undistort(intrinsics[camera], pixels), np.ones(len(ordered))])
        result = triangulate_keypoints(
            poses, rig.cameras, min_views=len(detections), confidence_threshold=0.0, refine=False,
        )
        valid = np.isfinite(result.points_3d).all(axis=1)
        points = result.points_3d[valid].astype(np.float64)
        if len(points) < 6:
            continue
        reference = board_points[ordered[valid]]
        ours = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        truth = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=-1)
        triangle = np.triu_indices(len(points), 1)
        errors.append(np.abs(ours[triangle] - truth[triangle]))
        reprojections.extend(result.reprojection_error[valid][np.isfinite(result.reprojection_error[valid])])
        used += 1
    if not errors:
        return {"frames": 0, "pairs": 0, "median_mm": None, "p90_mm": None,
                "reprojection_median_px": None, "reprojection_p90_px": None}
    combined = np.concatenate(errors)
    return {
        "frames": used,
        "pairs": int(len(combined)),
        "median_mm": round(float(np.median(combined)), 6),
        "p90_mm": round(float(np.percentile(combined, 90)), 6),
        "reprojection_median_px": round(float(np.median(reprojections)), 6),
        "reprojection_p90_px": round(float(np.percentile(reprojections, 90)), 6),
    }


def calibrate_detections(
    detections: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    intrinsics: Mapping[str, IntrinsicFit],
    image_sizes: Mapping[str, Tuple[int, int]],
    spec: BoardSpec,
    *,
    anchor: Optional[str] = None,
    holdout: float = 0.2,
    skip_joint: bool = False,
    world_unit: str = "mm",
    sync_mode: str = "hardware",
) -> CalibrationResult:
    """Refine a rig from already-detected corners (the image-independent core)."""
    names = list(detections)
    if len(names) < 2:
        raise ValueError("at least two cameras are needed for a calibration")
    anchor = anchor or names[0]
    if anchor not in detections:
        raise ValueError(f"anchor camera '{anchor}' is not one of {names}")
    points = spec.points()
    common = sorted(set.intersection(*(set(entries) for entries in detections.values())))
    if len(common) < 4:
        raise ValueError(f"at least four common frames are needed; got {len(common)}")

    extrinsics_all = initial_extrinsics(detections, intrinsics, points, common, anchor)
    poses_all = _board_poses(detections, intrinsics, points, common, extrinsics_all, anchor)
    frames_all = sorted(poses_all)
    if not frames_all:
        raise ValueError("the anchor camera has no usable frame for a board pose")

    if skip_joint:
        final_extrinsics = initial_extrinsics(detections, intrinsics, points, frames_all, anchor)
        final_poses = poses_all
        metrics = {
            "board_all": board_metrics(
                _rig_with(final_extrinsics, intrinsics, image_sizes, world_unit, sync_mode),
                detections, intrinsics, frames_all, points,
            ),
            "joint": False,
        }
    else:
        split = max(1, min(len(frames_all) - 1, int(round(len(frames_all) * (1.0 - holdout))))) if len(frames_all) > 4 else len(frames_all)
        fit_frames, holdout_frames = frames_all[:split], frames_all[split:]
        provisory_extrinsics = initial_extrinsics(detections, intrinsics, points, fit_frames, anchor)
        provisory_poses = _board_poses(detections, intrinsics, points, fit_frames, provisory_extrinsics, anchor)
        observations = _observations(detections, intrinsics, [frame for frame in fit_frames if frame in provisory_poses])
        rig = _rig_with({name: (np.eye(3), np.zeros(3)) for name in names},
                        intrinsics, image_sizes, world_unit, sync_mode)
        joint = bundle_adjust(
            points, observations, rig.cameras, provisory_extrinsics, provisory_poses,
            anchor=anchor, loss="cauchy", f_scale=5.0, max_nfev=300,
        )
        fit_rig = _rig_with(joint.extrinsics, intrinsics, image_sizes, world_unit, sync_mode)
        metrics = {
            "board_fit": board_metrics(fit_rig, detections, intrinsics, fit_frames, points),
            "board_holdout": board_metrics(fit_rig, detections, intrinsics, holdout_frames, points),
            "joint": {"median_px": round(float(joint.median_px), 6), "nfev": int(joint.nfev),
                      "parameters": int(joint.parameter_count)},
            "split": {"fit": len(fit_frames), "holdout": len(holdout_frames)},
        }
        # Refit on every usable frame for the final calibration.
        all_observations = _observations(detections, intrinsics, [frame for frame in frames_all if frame in poses_all])
        joint_all = bundle_adjust(
            points, all_observations, rig.cameras, joint.extrinsics, poses_all,
            anchor=anchor, loss="cauchy", f_scale=5.0, max_nfev=300,
        )
        final_extrinsics = joint_all.extrinsics
        final_poses = joint_all.board_poses
        metrics["board_all"] = board_metrics(
            _rig_with(final_extrinsics, intrinsics, image_sizes, world_unit, sync_mode),
            detections, intrinsics, frames_all, points,
        )

    cameras = {
        name: CameraCalibration(
            name, intrinsics[name].matrix, final_extrinsics[name][0], final_extrinsics[name][1],
            image_size=image_sizes.get(name), world_unit=world_unit,
        )
        for name in names
    }
    return CalibrationResult(
        cameras=cameras, extrinsics=final_extrinsics, intrinsics=dict(intrinsics),
        world_unit=world_unit, sync_mode=sync_mode, metrics=metrics,
    )


def write_camera_rig(
    path: Path,
    rig: CalibrationResult,
    *,
    force: bool = False,
) -> None:
    """Write and read back a ``cameras.yaml``; refuse to clobber without ``force``."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "world_unit": rig.world_unit,
        "sync": {"mode": rig.sync_mode},
        "cameras": {
            name: {
                "image_size": list(camera.image_size) if camera.image_size else None,
                "intrinsic": camera.intrinsic_matrix.tolist(),
                "rotation": camera.rotation_matrix.tolist(),
                "translation": camera.translation_vector.reshape(-1).tolist(),
                "distortion": {
                    "model": rig.intrinsics[name].model,
                    "coefficients": np.asarray(rig.intrinsics[name].coefficients).reshape(-1).tolist(),
                },
            }
            for name, camera in rig.cameras.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    try:
        load_camera_rig(temporary)
    except CalibrationError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)


def _parse_views(values: Sequence[str]) -> List[View]:
    views, seen = [], set()
    for item in values:
        camera, separator, path = str(item).partition("=")
        if not separator or not camera.strip() or not path.strip():
            raise ValueError(f"--view entries must look like camera=path, got '{item}'")
        camera = camera.strip()
        if camera in seen:
            raise ValueError(f"duplicate camera '{camera}' in --view")
        seen.add(camera)
        views.append(View(camera, Path(path.strip()).expanduser()))
    if len(views) < 2:
        raise ValueError("at least two --view entries are needed")
    return views


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli calibrate",
        description="Calibrate a multi-camera rig from ChArUco frames or videos.",
        epilog=(
            "Each view is either a folder of images or a video file. The command writes "
            "<output> only after a temporal holdout check and refuses to overwrite an "
            "existing calibration unless --force is given."
        ),
    )
    parser.add_argument("--view", action="append", default=[], required=True,
                        help="camera=path to a folder of calibration images or a video (repeatable)")
    parser.add_argument("--output", default="configs/cameras.yaml", help="output cameras.yaml")
    parser.add_argument("--anchor", help="anchor camera (default: the first --view)")
    parser.add_argument("--board-squares", default="10x7", help="board squares, e.g. 10x7")
    parser.add_argument("--square-size", type=float, default=22.0, help="square side in millimetres")
    parser.add_argument("--marker-size", type=float, default=16.5, help="ArUco marker side in millimetres")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--frame-step", type=int, default=0,
                        help="sample every N frames of a video (0: 1 for folders, 30 for videos)")
    parser.add_argument("--max-frames", type=int, default=300, help="maximum sampled frames per view")
    parser.add_argument("--min-corners", type=int, default=12, help="minimum corners per usable frame")
    parser.add_argument("--intrinsic-model", default="auto", choices=["auto", "none", "brown", "fisheye"])
    parser.add_argument("--intrinsic-frames", type=int, default=60,
                        help="frames used to fit intrinsics (OpenCV gets very slow beyond ~60)")
    parser.add_argument("--intrinsics-from", help="reuse the intrinsics and lens model of an existing cameras.yaml")
    parser.add_argument("--holdout", type=float, default=0.2, help="fraction of frames held out for validation")
    parser.add_argument("--skip-joint", action="store_true", help="use the PnP median without the bundle adjustment")
    parser.add_argument("--world-unit", default="mm", choices=["mm", "cm", "m"])
    parser.add_argument("--sync", default="hardware", choices=["hardware", "software", "unknown"])
    parser.add_argument("--report", help="optional JSON report path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing cameras.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        views = _parse_views(args.view)
        squares = str(args.board_squares).lower().split("x")
        if len(squares) != 2:
            raise ValueError("--board-squares must look like 10x7")
        spec = BoardSpec(int(squares[0]), int(squares[1]), args.square_size, args.marker_size,
                         args.dictionary.upper())
        print(f"detecting {len(views)} views on a {spec.squares_x}x{spec.squares_y} board ...", flush=True)
        detections, sizes = detect_views(
            views, spec, frame_step=args.frame_step, max_frames=args.max_frames,
            min_corners=args.min_corners,
        )
        for view in views:
            print(f"  {view.camera}: {len(detections[view.camera])} usable frames", flush=True)
        common = sorted(set.intersection(*(set(entries) for entries in detections.values())))
        print(f"  {len(common)} common frames", flush=True)

        if args.intrinsics_from:
            existing = load_camera_rig(Path(args.intrinsics_from).expanduser())
            intrinsics = {}
            for view in views:
                if view.camera not in existing.cameras:
                    raise ValueError(f"'{view.camera}' is not in {args.intrinsics_from}")
                model, coefficients = existing.distortion.get(view.camera, ("none", np.zeros(0)))
                intrinsics[view.camera] = IntrinsicFit(
                    existing.cameras[view.camera].intrinsic_matrix, model, np.asarray(coefficients)
                )
            print(f"reusing intrinsics from {args.intrinsics_from}", flush=True)
        else:
            intrinsics = {
                view.camera: fit_intrinsics(
                    detections[view.camera], sizes[view.camera], spec,
                    args.intrinsic_model, max_frames=args.intrinsic_frames,
                )
                for view in views
            }
            for view in views:
                fit = intrinsics[view.camera]
                print(f"  {view.camera}: {fit.model} | RMS {fit.rms:.3f} px | f {fit.matrix[0, 0]:.2f}",
                      flush=True)

        result = calibrate_detections(
            detections, intrinsics, sizes, spec,
            anchor=args.anchor or views[0].camera,
            holdout=args.holdout,
            skip_joint=args.skip_joint,
            world_unit=args.world_unit,
            sync_mode=args.sync,
        )
        write_camera_rig(Path(args.output), result, force=args.force)
        metrics = result.metrics
        for section in ("board_fit", "board_holdout", "board_all"):
            if section in metrics and metrics[section]["median_mm"] is not None:
                row = metrics[section]
                print(f"{section}: {row['frames']} frames | shape {row['median_mm']:.4f} mm "
                      f"| reprojection {row['reprojection_median_px']:.3f} px")
        print(f"wrote {args.output}", flush=True)
        if args.report:
            report = {
                "schema": "calibrate/1",
                "board": {
                    "squares": [spec.squares_x, spec.squares_y],
                    "square_mm": spec.square_mm,
                    "marker_mm": spec.marker_mm,
                    "dictionary": spec.dictionary,
                },
                "world_unit": result.world_unit,
                "sync": result.sync_mode,
                "frames": {"common": len(common), **metrics.get("split", {})},
                "intrinsics": {
                    name: {"model": fit.model, "rms_px": fit.rms,
                           "intrinsic": fit.matrix.tolist(),
                           "coefficients": np.asarray(fit.coefficients).reshape(-1).tolist()}
                    for name, fit in intrinsics.items()
                },
                "extrinsics": {
                    name: {"rotation": rotation.tolist(), "translation": translation.tolist()}
                    for name, (rotation, translation) in result.extrinsics.items()
                },
                "metrics": metrics,
            }
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"wrote {args.report}", flush=True)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, CalibrationError, RuntimeError) as error:
        print(f"calibrate failed: {error}")
        return 1
