"""Constant-velocity Kalman filters for optional boxes and primary keypoints."""

from typing import Sequence

import numpy as np


def xyxy_to_xyah(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    width, height = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5, width / height, height], dtype=np.float64)


def xyah_to_xyxy(value: Sequence[float]) -> np.ndarray:
    center_x, center_y, aspect, height = map(float, value[:4])
    height = max(height, 1e-3)
    width = max(aspect, 1e-3) * height
    return np.asarray([
        center_x - width * 0.5,
        center_y - height * 0.5,
        center_x + width * 0.5,
        center_y + height * 0.5,
    ], dtype=np.float32)


class KalmanBoxFilter:
    """Track center, aspect ratio, height and their frame-rate-scaled velocities."""

    def __init__(self, bbox_xyxy: Sequence[float]) -> None:
        measurement = xyxy_to_xyah(bbox_xyxy)
        self.mean = np.zeros((8,), dtype=np.float64)
        self.mean[:4] = measurement
        h = max(measurement[3], 1.0)
        self.covariance = np.diag([
            (0.05 * h) ** 2,
            (0.05 * h) ** 2,
            0.01**2,
            (0.05 * h) ** 2,
            (0.10 * h) ** 2,
            (0.10 * h) ** 2,
            0.02**2,
            (0.10 * h) ** 2,
        ])

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def predict(self, dt: int = 1) -> np.ndarray:
        dt = max(int(dt), 1)
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4, dtype=np.float64) * dt
        height = max(float(self.mean[3]), 1.0)
        position_noise = np.asarray([0.04 * height, 0.04 * height, 0.01, 0.04 * height]) * np.sqrt(dt)
        velocity_noise = np.asarray([0.02 * height, 0.02 * height, 0.005, 0.02 * height]) * np.sqrt(dt)
        process_noise = np.diag(np.concatenate([position_noise**2, velocity_noise**2]))
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.mean[2] = max(float(self.mean[2]), 1e-3)
        self.mean[3] = max(float(self.mean[3]), 1e-3)
        return self.bbox

    def update(self, bbox_xyxy: Sequence[float]) -> None:
        measurement = xyxy_to_xyah(bbox_xyxy)
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[:, :4] = np.eye(4, dtype=np.float64)
        height = max(float(self.mean[3]), 1.0)
        measurement_noise = np.diag([
            (0.05 * height) ** 2,
            (0.05 * height) ** 2,
            0.01**2,
            (0.05 * height) ** 2,
        ])
        residual = measurement - observation @ self.mean
        innovation_covariance = observation @ self.covariance @ observation.T + measurement_noise
        gain = np.linalg.solve(
            innovation_covariance,
            observation @ self.covariance,
        ).T
        self.mean = self.mean + gain @ residual
        identity = np.eye(8, dtype=np.float64)
        residual_map = identity - gain @ observation
        self.covariance = (
            residual_map @ self.covariance @ residual_map.T
            + gain @ measurement_noise @ gain.T
        )
        self.mean[2] = max(float(self.mean[2]), 1e-3)
        self.mean[3] = max(float(self.mean[3]), 1e-3)

    def apply_affine(self, matrix: np.ndarray) -> None:
        """Warp the current state by a 2x3 camera-motion transform."""
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (2, 3) or not np.isfinite(transform).all():
            raise ValueError("camera-motion matrix must be finite with shape (2, 3)")
        box = self.bbox.astype(np.float64)
        corners = np.asarray([
            [box[0], box[1], 1.0],
            [box[2], box[1], 1.0],
            [box[2], box[3], 1.0],
            [box[0], box[3], 1.0],
        ])
        moved = corners @ transform.T
        warped_box = [moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()]
        old_velocity = self.mean[4:6].copy()
        self.mean[:4] = xyxy_to_xyah(warped_box)
        self.mean[4:6] = transform[:, :2] @ old_velocity
        self.mean[6] *= max(float(np.linalg.norm(transform[:, 0])), 1e-3)
        self.mean[7] *= max(float(np.linalg.norm(transform[:, 1])), 1e-3)
        self.covariance[:4, :4] *= 1.25


class KalmanKeypointFilter:
    """Independent constant-velocity Kalman filters for each skeleton joint."""

    def __init__(self, keypoints: np.ndarray, confidence_threshold: float = 0.25, scale: float = 1.0):
        points = np.asarray(keypoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("keypoints must have shape (K, 2) or (K, 3)")
        self.keypoint_dims = int(points.shape[1])
        self.mean = np.zeros((len(points), 4), dtype=np.float64)
        self.covariance = np.tile(np.diag([25.0, 25.0, 100.0, 100.0]), (len(points), 1, 1))
        self.initialized = np.zeros((len(points),), dtype=bool)
        self.confidence = np.zeros((len(points),), dtype=np.float64)
        self.update(points, confidence_threshold, scale)

    @staticmethod
    def _confidence(points: np.ndarray) -> np.ndarray:
        if points.shape[1] == 3:
            return np.clip(points[:, 2], 0.0, 1.0)
        return np.ones((len(points),), dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        return self.mean[:, :2].astype(np.float32, copy=True)

    def as_keypoints(self) -> np.ndarray:
        result = np.zeros((len(self.mean), 3), dtype=np.float32)
        result[:, :2] = self.mean[:, :2].astype(np.float32)
        result[:, 2] = self.confidence.astype(np.float32)
        result[~self.initialized, :2] = 0.0
        result[~self.initialized, 2] = 0.0
        return result

    def predict(self, dt: int = 1, scale: float = 1.0) -> np.ndarray:
        dt = max(int(dt), 1)
        transition = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        extent = max(float(scale), 1.0)
        process_noise = np.diag([
            (0.015 * extent) ** 2 * dt,
            (0.015 * extent) ** 2 * dt,
            (0.005 * extent) ** 2 * dt,
            (0.005 * extent) ** 2 * dt,
        ])
        for index in np.flatnonzero(self.initialized):
            self.mean[index] = transition @ self.mean[index]
            covariance = self.covariance[index]
            self.covariance[index] = transition @ covariance @ transition.T + process_noise
        return self.positions

    def update(self, keypoints: np.ndarray, confidence_threshold: float = 0.25, scale: float = 1.0) -> None:
        points = np.asarray(keypoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] != len(self.mean) or points.shape[1] not in (2, 3):
            raise ValueError("keypoint count must stay constant within a tracker sequence")
        confidence = self._confidence(points)
        visible = confidence >= float(confidence_threshold)
        observation = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        extent = max(float(scale), 1.0)
        identity = np.eye(4, dtype=np.float64)
        for index in np.flatnonzero(visible):
            measurement = points[index, :2]
            score = max(float(confidence[index]), 0.05)
            measurement_noise = np.eye(2, dtype=np.float64) * ((0.025 * extent / score) ** 2)
            if not self.initialized[index]:
                self.mean[index, :2] = measurement
                self.mean[index, 2:] = 0.0
                self.covariance[index] = np.diag([
                    measurement_noise[0, 0],
                    measurement_noise[1, 1],
                    (0.10 * extent) ** 2,
                    (0.10 * extent) ** 2,
                ])
                self.initialized[index] = True
            else:
                covariance = self.covariance[index]
                innovation = measurement - observation @ self.mean[index]
                innovation_covariance = observation @ covariance @ observation.T + measurement_noise
                gain = np.linalg.solve(innovation_covariance, observation @ covariance).T
                self.mean[index] += gain @ innovation
                residual_map = identity - gain @ observation
                self.covariance[index] = (
                    residual_map @ covariance @ residual_map.T
                    + gain @ measurement_noise @ gain.T
                )
            self.confidence[index] = float(confidence[index])

    def snapshot(self):
        return (
            self.mean.copy(),
            self.covariance.copy(),
            self.initialized.copy(),
            self.confidence.copy(),
        )

    def restore(self, snapshot) -> None:
        self.mean, self.covariance, self.initialized, self.confidence = (
            item.copy() for item in snapshot
        )

    def apply_affine(self, matrix: np.ndarray) -> None:
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (2, 3) or not np.isfinite(transform).all():
            raise ValueError("camera-motion matrix must be finite with shape (2, 3)")
        self.mean[:, :2] = self.mean[:, :2] @ transform[:, :2].T + transform[:, 2]
        self.mean[:, 2:] = self.mean[:, 2:] @ transform[:, :2].T
        state_transform = np.zeros((4, 4), dtype=np.float64)
        state_transform[:2, :2] = transform[:, :2]
        state_transform[2:, 2:] = transform[:, :2]
        for index in np.flatnonzero(self.initialized):
            self.covariance[index] = (
                state_transform @ self.covariance[index] @ state_transform.T
            )


__all__ = ["KalmanBoxFilter", "KalmanKeypointFilter", "xyxy_to_xyah", "xyah_to_xyxy"]
