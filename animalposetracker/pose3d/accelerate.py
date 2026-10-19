"""Batched triangulation with JAX as an optional backend.

The numpy path is the reference implementation and runs everywhere. The JAX path exists for
long recordings, where the point count runs into the millions and one batched SVD over the
whole segment beats both a per-point loop and a chunked numpy version. Both paths are
compared against each other in the tests.

``animalpose-cli pose3d`` does not call this yet -- it still triangulates per keypoint,
because the RANSAC step picks a different camera subset for each one. This module is the
seat for the batched path, so a future long-video mode has somewhere to land.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["jax_available", "triangulate_batch"]


def jax_available() -> bool:
    """True when jax imports cleanly (the ``pose3d-jax`` extra is installed)."""
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    return True


def _design(points: np.ndarray, projection: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Build the ``(K, 2C, 4)`` DLT design matrices.

    ``points`` is ``(K, C, 2)``, ``projection`` is ``(C, 3, 4)`` and ``weights`` is
    ``(K, C)``. Each camera contributes the two rows

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


def _triangulate_jax(points, projection, weights) -> np.ndarray:
    import jax.numpy as jnp

    matrix = jnp.asarray(_design(points, projection, weights))
    _, _, vh = jnp.linalg.svd(matrix, full_matrices=False)
    homogeneous = np.asarray(vh[:, -1])
    return _to_cartesian(homogeneous)


def triangulate_batch(
    points: np.ndarray,
    projection: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Triangulate ``K`` points from up to ``C`` cameras in one call.

    Args:
        points: ``(K, C, 2)`` image coordinates, one row per camera.
        projection: ``(C, 3, 4)`` projection matrices, in the same camera order.
        weights: ``(K, C)`` per-observation confidence; ones when omitted.
        backend: ``"auto"`` uses JAX when it imports, ``"numpy"`` forces the baseline and
            ``"jax"`` forces the accelerator (and fails if it is missing).

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

    if backend not in ("auto", "numpy", "jax"):
        raise ValueError("backend must be 'auto', 'numpy' or 'jax'")
    if backend == "jax" and not jax_available():
        raise RuntimeError("backend='jax' needs the pose3d-jax extra installed")
    use_jax = backend == "jax" or (backend == "auto" and jax_available())
    return _triangulate_jax(points, projection, weights) if use_jax \
        else _triangulate_numpy(points, projection, weights)
