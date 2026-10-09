"""Affine crop geometry shared by top-down data and inference paths."""

from typing import Tuple

import cv2
import numpy as np


def _rotate_point(point: np.ndarray, angle_radians: float) -> np.ndarray:
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    return np.asarray(
        [point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine],
        dtype=np.float32,
    )

def _third_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = first - second
    return second + np.asarray([-direction[1], direction[0]], dtype=np.float32)

def _topdown_warp_matrix(center, scale, rotation, output_size):
    """Port MMPose's ``get_warp_matrix`` for the default non-UDP affine transform."""
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    output_width, output_height = map(int, output_size)
    angle = np.deg2rad(float(rotation))
    source_direction = _rotate_point(np.asarray([-scale[0] * 0.5, 0.0]), angle)
    target_direction = np.asarray([-output_width * 0.5, 0.0], dtype=np.float32)

    source = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    source[2] = _third_point(source[0], source[1])

    target_center = np.asarray([output_width * 0.5, output_height * 0.5], dtype=np.float32)
    target = np.zeros((3, 2), dtype=np.float32)
    target[0] = target_center
    target[1] = target_center + target_direction
    target[2] = _third_point(target[0], target[1])
    return cv2.getAffineTransform(source, target)

def _fix_aspect_ratio(scale: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    width, height = map(float, scale)
    output_width, output_height = input_size
    aspect_ratio = output_width / output_height
    if width > height * aspect_ratio:
        height = width / aspect_ratio
    else:
        width = height * aspect_ratio
    return np.asarray([width, height], dtype=np.float32)
