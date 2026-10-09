"""AnimalPoseTracker's independent model-training framework.

Importing this package does not import PyTorch. Training-only dependencies are
loaded by the modules that need them, so the GUI and inference tools can run in
environments without a training installation.
"""

from .config import TrainingConfig

__all__ = ["TrainingConfig"]
