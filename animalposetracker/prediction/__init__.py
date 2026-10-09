"""Offline prediction APIs for images, videos, and dataset splits."""

from .backends import ModelArtifactBackend, PREDICT_FORMATS

__all__ = ["ModelArtifactBackend", "PREDICT_FORMATS"]
