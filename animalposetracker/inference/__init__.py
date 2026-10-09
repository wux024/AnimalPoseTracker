"""Real-time, frame-oriented pose inference APIs, loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "InferenceEngine": (".inferencer", "InferenceEngine"),
    "PLATFORM": (".constant", "PLATFORM"),
    "ENGINEtoDEVICE": (".constant", "ENGINEtoDEVICE"),
    "ENGINEtoBackend": (".constant", "ENGINEtoBackend"),
    "OpenCV_TARGETS": (".constant", "OpenCV_TARGETS"),
    "DetectorBox": (".stream_inference", "DetectorBox"),
    "DetectorProvider": (".stream_inference", "DetectorProvider"),
    "StreamInferencePipeline": (".stream_inference", "StreamInferencePipeline"),
    "StreamInputConfig": (".stream_inference", "StreamInputConfig"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
