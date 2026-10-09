"""Pose model validation metrics and validators, loaded on demand."""

from importlib import import_module

_METRIC_NAMES = {
    "IOU_THRESHOLDS", "box_iou", "build_coco_ground_truth", "evaluate_coco_keypoints",
    "keypoint_error_metrics", "keypoint_error_metrics_from_coco_matches", "keypoint_oks",
    "match_predictions", "plot_confusion_matrix", "plot_precision_recall", "summarize_ap",
    "update_confusion_matrix",
}
_EXPORTS = {name: (".metrics", name) for name in _METRIC_NAMES}
_EXPORTS.update({
    "PoseDetectionValidator": (".animalrtpose", "PoseDetectionValidator"),
    "SimCCPoseValidator": (".topdown", "SimCCPoseValidator"),
})
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
