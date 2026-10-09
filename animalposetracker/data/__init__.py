"""Dataset readers and annotation format adapters."""

__all__ = ["PoseTextDataset", "pose_collate", "build_pose_dataloaders"]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module

        pose = import_module(".pose", __name__)
        return getattr(pose, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
