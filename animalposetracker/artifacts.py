"""Shared model-artifact format and metadata discovery."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


EXPORT_FORMATS = (
    "torchscript", "onnx", "openvino", "engine", "coreml", "saved_model",
    "pb", "tflite", "edgetpu", "tfjs", "paddle", "mnn", "ncnn", "imx", "rknn",
)

PREDICT_FORMATS = (
    "pt", "torchscript", "onnx", "openvino", "engine", "coreml", "saved_model",
    "pb", "tflite", "edgetpu", "paddle", "mnn", "ncnn", "imx", "rknn", "triton",
)


def read_artifact_metadata(model_path: Path) -> Dict[str, Any]:
    """Read a model sidecar when present."""
    model_path = Path(model_path)
    candidates = [
        model_path.with_suffix(model_path.suffix + ".json"),
        model_path.with_suffix(".json"),
        model_path / "metadata.json" if model_path.is_dir() else None,
        model_path / "metadata.yaml" if model_path.is_dir() else None,
        model_path.parent / "metadata.json" if model_path.is_dir() else None,
        model_path.parent / "metadata.yaml" if model_path.is_dir() else None,
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".yaml":
            import yaml

            with candidate.open("r", encoding="utf-8") as stream:
                payload = yaml.safe_load(stream) or {}
        else:
            with candidate.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        if isinstance(payload, dict):
            return payload
    return {}


def detect_artifact_format(model_path, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Resolve a supported model artifact format from metadata or its path."""
    reference = str(model_path)
    if reference.startswith(("http://", "grpc://")):
        return "triton"
    model_path = Path(model_path)
    metadata = metadata or read_artifact_metadata(model_path)
    declared = str(metadata.get("format", "")).strip().lower()
    aliases = {
        "tensorrt": "engine", "trt": "engine", "mlmodel": "coreml", "mlpackage": "coreml"
    }
    declared = aliases.get(declared, declared)
    if declared in EXPORT_FORMATS:
        return declared

    suffix = model_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return "pt"
    if suffix in {".torchscript", ".jit"}:
        return "torchscript"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".engine":
        return "engine"
    if suffix in {".mlmodel", ".mlpackage"}:
        return "coreml"
    if suffix == ".xml" or model_path.is_dir() and list(model_path.glob("*.xml")):
        return "openvino"
    if suffix == ".pb":
        return "pb"
    if suffix == ".tflite":
        return "edgetpu" if "edgetpu" in model_path.name.lower() else "tflite"
    if suffix == ".mnn":
        return "mnn"
    if suffix == ".rknn":
        return "rknn"
    if model_path.is_dir():
        name = model_path.name.lower()
        if name.endswith("_saved_model") or (model_path / "saved_model.pb").is_file():
            return "saved_model"
        if name.endswith("_paddle_model") or list(model_path.glob("*.pdmodel")) or list(model_path.glob("*.json")):
            return "paddle"
        if name.endswith("_ncnn_model") or list(model_path.glob("*.param")):
            return "ncnn"
        if name.endswith("_imx_model") or list(model_path.glob("*.onnx")):
            return "imx" if name.endswith("_imx_model") else "onnx"
        if name.endswith("_rknn_model") or list(model_path.glob("*.rknn")):
            return "rknn"
    raise ValueError(
        f"Cannot determine a supported pose-model format from {model_path}. "
        f"Supported project prediction formats: {', '.join(PREDICT_FORMATS)}"
    )


__all__ = [
    "EXPORT_FORMATS", "PREDICT_FORMATS", "read_artifact_metadata", "detect_artifact_format"
]
