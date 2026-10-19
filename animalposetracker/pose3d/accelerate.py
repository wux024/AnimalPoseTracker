"""Batched triangulation with three interchangeable backends.

``numpy``   — the reference. One batched SVD over the whole ``(K, 2C, 4)`` design matrix,
              no Python loop over points. Runs anywhere numpy does.
``jax-cpu`` — the same maths through ``jax.numpy``, pinned to the CPU device. Faster than
              numpy only when several cores are available and the batch is large.
``jax-gpu`` — the same, pinned to the GPU. This is the one that matters for hour-long
              recordings, where the point count runs into the millions.

``auto`` picks ``jax-gpu`` when a GPU is visible, ``jax-cpu`` when jax is installed without
one, and ``numpy`` when jax is absent. Every path is compared against the per-point
implementation in the tests.

``animalpose-cli pose3d`` does not call this yet -- it still triangulates per keypoint,
because the RANSAC step picks a different camera subset for each one. This module is the
seat for the batched path, so a future long-video mode has somewhere to land.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["available_backends", "jax_available", "triangulate_batch"]

_BACKENDS = ("auto", "numpy", "jax-cpu", "jax-gpu")


def jax_available(device: Optional[str] = None) -> bool:
    """True when jax imports and, if ``device`` is given, that device exists."""
    try:
        import jax
    except Exception:
        return False
    if device is None:
        return True
    try:
        jax.devices(device)
    except Exception:
        return False
    return True


def available_backends() -> dict:
    """Describe what this machine can run, for the CLI banner and the tests."""
    has_jax = jax_available()
    return {
        "numpy": True,
        "jax-cpu": has_jax and jax_available("cpu"),
        "jax-gpu": has_jax and jax_available("gpu"),
    }


def _design(points: np.ndarray, projection: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Build the ``(K, 2C, 4)`` DLT design matrices.

    ``points`` is ``(K, C, 2)`` in normalised image coordinates, ``projection`` is
    ``(C, 3, 4)`` and ``weights`` is ``(K, C)``. Each camera contributes the two rows

        x * P[2] - P[0]
        y * P[2] - P[1]

    scaled by ``sqrt(weight)``, exactly as the per-point implementation does.
    """
    x = points[..., 0][..., None]
    y = points[..., 1][..., None]
    p = projection[None]
    scale = np.sqrt(np.maximum(weights, 1e-6))[..., None]
    row_x = (x * p[..., 2, :] - p[..., 0, :]) * scale
    row_y = (y * p[..., 2, :] - p[..., 1, :]) * scale
    stacked = np.stack([row_x, row_y], axis=2)
    return stacked.reshape(points.shape[0], -1, 4)


def _to_cartesian(homogeneous: np.ndarray) -> np.ndarray:
    result = np.full((homogeneous.shape[0], 3), np.nan, dtype=np.float64)
    valid = np.abs(homogeneous[:, 3]) >= 1e-12
    result[valid] = homogeneous[valid, :3] / homogeneous[valid, 3:4]
    result[~np.isfinite(result).all(axis=1)] = np.nan
    return result


def _triangulate_numpy(points, projection, weights) -> np.ndarray:
    matrix = _design(points, projection, weights)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return _to_cartesian(vh[:, -1])


def _triangulate_jax(points, projection, weights, device: str) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    matrix = _design(points, projection, weights)
    with jax.default_device(jax.devices(device)[0]):
        matrix = jnp.asarray(matrix)
        _, _, vh = jnp.linalg.svd(matrix, full_matrices=False)
        homogeneous = np.asarray(vh[:, -1])
    return _to_cartesian(homogeneous)


def _resolve(backend: str) -> str:
    """Turn ``auto`` into a concrete backend, and validate the rest."""
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(_BACKENDS)}")
    if backend != "auto":
        return backend
    if not jax_available():
        return "numpy"
    return "jax-gpu" if jax_available("gpu") else "jax-cpu"


def triangulate_batch(
    points: np.ndarray,
    projection: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Triangulate ``K`` points from up to ``C`` cameras in one call.

    Args:
        points: ``(K, C, 2)`` normalised image coordinates, one row per camera.
        projection: ``(C, 3, 4)`` projection matrices, in the same camera order.
        weights: ``(K, C)`` per-observation confidence; ones when omitted.
        backend: ``auto``, ``numpy``, ``jax-cpu`` or ``jax-gpu``.

    Returns:
        ``(K, 3)`` world points. Rows that could not be solved are ``nan``, matching the
        ``None`` return of the per-point path.
    """
    points = np.asarray(points, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("points must be (K, C, 2)")
    if projection.shape[-2:] != (3, 4):
        raise ValueError("projection must be (C, 3, 4)")
    if weights is None:
        weights = np.ones(points.shape[:2], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    resolved = _resolve(backend)
    if resolved == "numpy":
        return _triangulate_numpy(points, projection, weights)
    if not jax_available():
        raise RuntimeError(f"backend='{resolved}' needs the pose3d-jax extra installed")
    device = "gpu" if resolved == "jax-gpu" else "cpu"
    if not jax_available(device):
        raise RuntimeError(f"backend='{resolved}' needs a jax {device.upper()} device")
    return _triangulate_jax(points, projection, weights, device)
