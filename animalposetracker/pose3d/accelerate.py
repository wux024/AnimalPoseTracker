"""Batched triangulation with three interchangeable backends.

``numpy``   — the reference and the default. One batched SVD over the whole ``(K, 2C, 4)``
              design matrix, no Python loop over points. Runs anywhere numpy does, and it
              won the local benchmark at clip-sized batches.
``numba``   — a ``prange`` loop with one SVD per point, compiled to machine code. Roughly
              three times faster than numpy past sixty thousand points, so it is the CPU
              answer for long recordings.
``jax-gpu`` — the same maths through ``jax.numpy`` on a GPU. This is the one that matters
              when the point count runs into the millions. A jax CPU backend was measured
              and dropped: XLA's CPU SVD came out slower than plain numpy.

``auto`` means ``numpy``: installing an accelerator never changes behaviour on its own, the
backend has to be asked for. Every path is compared against the per-point implementation in
the tests.

``animalpose-cli pose3d --backend <name>`` routes the DLT solve through this module.
The batched solver takes one camera set per call, so the CLI turns RANSAC off when a
backend is named: the per-point RANSAC path picks a different camera subset for every
joint and cannot be batched. Without ``--backend`` the per-point path stays the default.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["BACKENDS", "available_backends", "jax_available", "numba_available", "triangulate_batch"]

BACKENDS = ("auto", "numpy", "numba", "jax-gpu")

_BACKENDS = BACKENDS


def jax_available() -> bool:
    """True when jax imports with a visible GPU device (the ``pose3d-jax`` extra)."""
    try:
        import jax
    except Exception:
        return False
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


def numba_available() -> bool:
    """True when numba imports (the ``pose3d-numba`` extra is installed)."""
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def available_backends() -> dict:
    """Describe what this machine can run, for the CLI banner and the tests."""
    return {
        "numpy": True,
        "numba": numba_available(),
        "jax-gpu": jax_available(),
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


def _triangulate_numba(points, projection, weights) -> np.ndarray:
    from numba import njit, prange

    matrix = _design(points, projection, weights)

    @njit(cache=True, parallel=True)
    def _solve(design):  # pragma: no cover - compiled at runtime
        count = design.shape[0]
        solution = np.empty((count, 4), dtype=np.float64)
        for index in prange(count):
            # numba's np.linalg.svd takes a single matrix and no `full_matrices` keyword;
            # for a (2C, 4) design a full decomposition still returns a 4x4 vh.
            _u, _s, vh = np.linalg.svd(design[index])
            solution[index] = vh[-1]
        return solution

    return _to_cartesian(_solve(matrix))


def _triangulate_jax(points, projection, weights) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    matrix = _design(points, projection, weights)
    with jax.default_device(jax.devices("gpu")[0]):
        matrix = jnp.asarray(matrix)
        _, _, vh = jnp.linalg.svd(matrix, full_matrices=False)
        homogeneous = np.asarray(vh[:, -1])
    return _to_cartesian(homogeneous)


def _resolve(backend: str) -> str:
    """Turn ``auto`` into a concrete backend, and validate the rest."""
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(_BACKENDS)}")
    if backend == "auto":
        return "numpy"
    return backend


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
        backend: ``auto``, ``numpy``, ``numba`` or ``jax-gpu``.

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
    if resolved == "numba":
        if not numba_available():
            raise RuntimeError("backend='numba' needs the pose3d-numba extra installed")
        return _triangulate_numba(points, projection, weights)
    if not jax_available():
        raise RuntimeError(
            "backend='jax-gpu' needs the pose3d-jax extra and a visible GPU device"
        )
    return _triangulate_jax(points, projection, weights)
