"""Model-output decoding and optional pose-result processing, loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "PoseInstance": (".output_decoders", "PoseInstance"),
    "PoseOutputConfig": (".output_decoders", "PoseOutputConfig"),
    "RTMO_MMDEPLOY": (".output_decoders", "RTMO_MMDEPLOY"),
    "RTMPOSE_SIMCC": (".output_decoders", "RTMPOSE_SIMCC"),
    "SUPPORTED_OUTPUT_SCHEMAS": (".output_decoders", "SUPPORTED_OUTPUT_SCHEMAS"),
    "YOLO_POSE_END2END": (".output_decoders", "YOLO_POSE_END2END"),
    "YOLO_POSE_NMS": (".output_decoders", "YOLO_POSE_NMS"),
    "YOLO_POSE_RAW": (".output_decoders", "YOLO_POSE_RAW"),
    "decode_pose_outputs": (".output_decoders", "decode_pose_outputs"),
    "normalize_output_tensors": (".output_decoders", "normalize_output_tensors"),
    "filter_pose_2d": (".filters", "filter_pose_2d"),
    "filter_pose_3d": (".filters", "filter_pose_3d"),
    "track_frame_predictions": (".workflows", "track_frame_predictions"),
    "filter_tracked_video_records": (".workflows", "filter_tracked_video_records"),
    "triangulate_multiview_predictions": (".workflows", "triangulate_multiview_predictions"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
