"""Optional frame-to-frame global camera motion compensation."""

from typing import Optional

import numpy as np


class CameraMotionEstimator:
    """Estimate an affine previous-frame to current-frame transform."""

    METHODS = {"sparseOptFlow", "orb", "sift", "ecc"}

    def __init__(self, method: str = "none") -> None:
        self.method = str(method)
        if self.method != "none" and self.method not in self.METHODS:
            raise ValueError(f"Unsupported camera-motion method {self.method!r}")
        self._previous_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._previous_gray = None

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return an identity transform for the first frame or failed estimates."""
        import cv2

        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image.copy()
        else:
            raise ValueError("frame must be a grayscale or BGR image")
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        if self._previous_gray is None:
            self._previous_gray = gray
            return identity

        previous = self._previous_gray
        self._previous_gray = gray
        try:
            transform = self._estimate_pair(previous, gray, cv2)
        except cv2.error:
            transform = None
        if transform is None or transform.shape != (2, 3) or not np.isfinite(transform).all():
            return identity
        return transform.astype(np.float32, copy=False)

    def _estimate_pair(self, previous: np.ndarray, current: np.ndarray, cv2):
        if self.method == "sparseOptFlow":
            source = cv2.goodFeaturesToTrack(
                previous, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=3
            )
            if source is None or len(source) < 3:
                return None
            target, status, _error = cv2.calcOpticalFlowPyrLK(previous, current, source, None)
            if target is None or status is None:
                return None
            keep = status.reshape(-1).astype(bool)
            if int(keep.sum()) < 3:
                return None
            transform, _inliers = cv2.estimateAffinePartial2D(
                source.reshape(-1, 2)[keep],
                target.reshape(-1, 2)[keep],
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            return transform

        if self.method in {"orb", "sift"}:
            detector = cv2.ORB_create(nfeatures=500) if self.method == "orb" else cv2.SIFT_create(nfeatures=500)
            keypoints_a, descriptors_a = detector.detectAndCompute(previous, None)
            keypoints_b, descriptors_b = detector.detectAndCompute(current, None)
            if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 3 or len(keypoints_b) < 3:
                return None
            norm = cv2.NORM_HAMMING if self.method == "orb" else cv2.NORM_L2
            matches = cv2.BFMatcher(norm).knnMatch(descriptors_a, descriptors_b, k=2)
            good = [first for pair in matches if len(pair) == 2 for first, second in [pair] if first.distance < 0.75 * second.distance]
            if len(good) < 3:
                return None
            points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good])
            points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good])
            transform, _inliers = cv2.estimateAffinePartial2D(
                points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            return transform

        # findTransformECC maps the current input back to the previous template;
        # invert it to get the transform used to warp track states forward.
        warp = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(
            previous,
            current,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
            None,
            1,
        )
        return cv2.invertAffineTransform(warp)


__all__ = ["CameraMotionEstimator"]
