"""SimCC keypoint target encoding for top-down pose data."""

from typing import Tuple

import numpy as np


class SimCCLabel:
    """Encode keypoints as the 1D Gaussian labels used by MMPose ``SimCCLabel``."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        sigma: float = 6.0,
        split_ratio: float = 2.0,
        normalize: bool = True,
    ) -> None:
        self.input_size = tuple(map(int, input_size))
        self.sigma = float(sigma)
        self.split_ratio = float(split_ratio)
        self.normalize = bool(normalize)
        if len(self.input_size) != 2 or min(self.input_size) < 1:
            raise ValueError("SimCC input_size must be a positive (width, height) pair")
        if self.sigma <= 0 or self.split_ratio <= 0:
            raise ValueError("SimCC sigma and split_ratio must be positive")

    def encode(self, keypoints: np.ndarray, visible: np.ndarray):
        """Return x/y labels and visibility weights for one cropped instance."""
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        weights = np.asarray(visible, dtype=np.float32).reshape(-1).copy()
        if len(weights) != len(keypoints):
            raise ValueError("SimCC keypoint coordinates and visibility have different lengths")

        width, height = self.input_size
        bins_x = int(np.around(width * self.split_ratio))
        bins_y = int(np.around(height * self.split_ratio))
        labels_x = np.zeros((len(keypoints), bins_x), dtype=np.float32)
        labels_y = np.zeros((len(keypoints), bins_y), dtype=np.float32)
        split_points = np.around(keypoints * self.split_ratio).astype(np.int64)
        x_grid = np.arange(bins_x, dtype=np.float32)
        y_grid = np.arange(bins_y, dtype=np.float32)
        radius = self.sigma * 3.0

        for index, (mu_x, mu_y) in enumerate(split_points):
            if weights[index] < 0.5:
                continue
            if mu_x < 0 or mu_y < 0 or mu_x >= bins_x or mu_y >= bins_y:
                weights[index] = 0.0
                continue
            left, top = mu_x - radius, mu_y - radius
            right, bottom = mu_x + radius + 1, mu_y + radius + 1
            if left >= bins_x or top >= bins_y or right < 0 or bottom < 0:
                weights[index] = 0.0
                continue
            labels_x[index] = np.exp(-((x_grid - mu_x) ** 2) / (2.0 * self.sigma**2))
            labels_y[index] = np.exp(-((y_grid - mu_y) ** 2) / (2.0 * self.sigma**2))

        if self.normalize:
            norm = self.sigma * np.sqrt(2.0 * np.pi)
            labels_x /= norm
            labels_y /= norm
        return labels_x, labels_y, weights
