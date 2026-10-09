"""Native PyTorch export worker for project pose models."""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from animalposetracker.nn.head import SimCCHead, YOLOPoseHead
from animalposetracker.export_backends import export_with_backend
from animalposetracker.prediction_backends import EXPORT_FORMATS
from animalposetracker.workflows import load_project_model, unique_output_directory


class _ExportAdapter(torch.nn.Module):
    """Expose only decoded pose outputs, excluding training-only auxiliary tensors."""

    def __init__(self, model, head_name: str):
        super().__init__()
        self.model = model
        self.head_name = head_name

    def forward(self, images):
        outputs = self.model(images)
        if self.head_name == "YOLOPoseHead":
            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if self.head_name == "SimCCHead":
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                raise TypeError("SimCC export expects two outputs: x and y distributions")
            return outputs[0], outputs[1]
        raise NotImplementedError(f"No export adapter exists for {self.head_name}")


def _make_argument_parser():
    parser = argparse.ArgumentParser(
        prog="animalposetracker-export",
        description="Export AnimalPoseTracker checkpoints without the YOLO CLI.",
    )
    parser.add_argument("--config", default="configs/other.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", help="Override the export directory")
    parser.add_argument(
        "--format",
        help="Explicit deployment format; may also be set as application.format in the project config",
    )
    return parser


def _export_onnx(adapter, dummy, path, context, values):
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("ONNX export requires the optional `animalposetracker[export]` dependencies") from exc

    head_name = context["head_name"]
    dynamic = bool(values.get("dynamic", False))
    batch_axis = {0: "batch"}
    dynamic_axes = {"images": dict(batch_axis)}
    if head_name == "YOLOPoseHead" and dynamic:
        dynamic_axes["images"].update({2: "height", 3: "width"})
        dynamic_axes["predictions"] = {0: "batch", 2: "anchors"}
    elif head_name == "SimCCHead" and dynamic:
        dynamic_axes["simcc_x"] = dict(batch_axis)
        dynamic_axes["simcc_y"] = dict(batch_axis)

    output_names = ["predictions"] if head_name == "YOLOPoseHead" else ["simcc_x", "simcc_y"]
    torch.onnx.export(
        adapter,
        dummy,
        str(path),
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic else None,
        opset_version=int(values.get("opset") or 17),
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    if bool(values.get("simplify", False)):
        try:
            from onnxsim import simplify
        except ImportError as exc:
            raise ImportError(
                "ONNX simplification requires the optional `animalposetracker[export]` dependencies"
            ) from exc
        simplified, valid = simplify(onnx_model)
        if not valid:
            raise RuntimeError("onnxsim reported that the simplified model is invalid")
        onnx.save(simplified, str(path))


def _resolve_export_format(requested, configured):
    """Resolve an explicitly selected export format without an implicit default."""
    value = requested if requested is not None else configured
    if value is None or not str(value).strip():
        raise ValueError(
            "No export format selected. Pass --format <format> or set application.format "
            "in configs/other.yaml. Training checkpoints are already saved as .pt."
        )
    return str(value).strip().lower()


def run(argv=None) -> int:
    args = _make_argument_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    model, config, context = load_project_model(config_path, args.weights)
    context["dataset_config_path"] = config.data
    values = context["other"]
    export_format = _resolve_export_format(args.format, values.get("format"))
    if export_format in {"tensorrt", "trt"}:
        export_format = "engine"
    if export_format not in EXPORT_FORMATS:
        raise NotImplementedError(
            f"Native export does not recognize {export_format!r}; supported targets: "
            f"{', '.join(EXPORT_FORMATS)}"
        )
    if bool(values.get("nms", False)):
        raise NotImplementedError("Embedding NMS in the exported graph is not implemented")
    format_options = {
        "torchscript": {"half", "optimize"},
        "onnx": {"dynamic", "half", "opset", "simplify"},
        "openvino": {"dynamic", "half", "opset", "simplify"},
        "engine": {"dynamic", "half", "opset", "simplify", "workspace"},
        "coreml": {"half"},
        "saved_model": {"opset", "simplify"},
        "pb": {"opset", "simplify"},
        "tflite": {"half", "opset", "simplify", "int8"},
        "edgetpu": {"opset", "simplify", "int8"},
        "tfjs": {"half", "opset", "simplify"},
        "paddle": set(),
        "mnn": {"half", "int8", "opset", "simplify"},
        "ncnn": {"half"},
        "imx": {"int8", "opset", "simplify"},
        "rknn": {"opset", "simplify"},
    }
    for option in ("half", "int8", "dynamic", "optimize"):
        if bool(values.get(option, False)) and option not in format_options[export_format]:
            raise ValueError(f"{option} is not supported for {export_format} export")
    if bool(values.get("keras", False)):
        raise NotImplementedError("Keras SavedModel export is not implemented; use format=saved_model")
    if values.get("opset") is not None and "opset" not in format_options[export_format]:
        raise ValueError(f"opset does not apply to {export_format} export")
    if bool(values.get("workspace", 0)) and "workspace" not in format_options[export_format]:
        raise ValueError("workspace is only meaningful for TensorRT export")
    if export_format == "engine" and bool(values.get("int8", False)):
        raise NotImplementedError("TensorRT INT8 requires a representative calibration loader")
    if export_format in {"torchscript", "coreml", "paddle", "ncnn"} and bool(values.get("simplify", False)):
        print(f"Note: simplify applies to ONNX graph conversion; {export_format} exports directly from PyTorch/TorchScript.")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        config.output_dir = (
            output_dir if output_dir.is_absolute()
            else context["project_dir"] / output_dir
        ).resolve()
    config.output_dir = unique_output_directory(
        config.output_dir,
        exist_ok=bool(values.get("exist_ok", False)),
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = context["weights"].stem
    path_suffixes = {
        "torchscript": f"{stem}.torchscript",
        "onnx": f"{stem}.onnx",
        "openvino": f"{stem}_openvino_model",
        "engine": f"{stem}.engine",
        "coreml": f"{stem}.mlpackage",
        "saved_model": f"{stem}_saved_model",
        "pb": f"{stem}.pb",
        "tflite": f"{stem}.tflite",
        "edgetpu": f"{stem}_edgetpu.tflite",
        "tfjs": f"{stem}_web_model",
        "paddle": f"{stem}_paddle_model",
        "mnn": f"{stem}.mnn",
        "ncnn": f"{stem}_ncnn_model",
        "imx": f"{stem}_imx_model",
        "rknn": f"{stem}_rknn_model",
    }
    output_path = output_dir / path_suffixes[export_format]

    half = bool(values.get("half", False))
    if export_format == "imx" and half:
        raise ValueError("Sony IMX500 export uses an INT8 quantized graph; half=true does not apply")
    if half and context["device"].type != "cuda" and export_format in {"torchscript", "onnx"}:
        raise ValueError(f"half=true requires a CUDA export device for {export_format}")
    dtype = torch.float16 if half and context["device"].type == "cuda" else torch.float32
    model = model.to(device=context["device"], dtype=dtype).eval()
    head = model.model[-1]
    if isinstance(head, YOLOPoseHead):
        head.export = True
        head.format = export_format
        head.dynamic = bool(values.get("dynamic", False))
    elif not isinstance(head, SimCCHead):
        raise NotImplementedError(f"Native export does not support head {type(head).__name__}")

    adapter = _ExportAdapter(model, context["head_name"]).eval()
    size = int(config.image_size)
    dummy = torch.zeros(
        1,
        3,
        size,
        size,
        device=context["device"],
        dtype=dtype,
    )

    metadata = {
        "format": export_format,
        "model_type": context["project"].get("model_type"),
        "model_scale": context["scale"],
        "head": context["head_name"],
        "input_size": [size, size],
        "input_name": "images",
        "input_tensor": "images:0",
        "input_layout": "NCHW",
        "input_dtype": "float32" if dtype == torch.float32 else "float16",
        "color_order": "RGB",
        "kpt_shape": list(model.kpt_shape),
        "weights": str(context["weights"]),
        "outputs": ["predictions"] if context["head_name"] == "YOLOPoseHead" else ["simcc_x", "simcc_y"],
    }
    if context["head_name"] == "SimCCHead":
        model_preprocessing = context["model_spec"].get("preprocessing") or {}
        metadata.update({
            "simcc_split_ratio": float(model.model[-1].simcc_split_ratio),
            "bbox_padding": float(model_preprocessing.get("bbox_padding", 1.25)),
            "pixel_mean": model_preprocessing.get("pixel_mean", [123.675, 116.28, 103.53]),
            "pixel_std": model_preprocessing.get("pixel_std", [58.395, 57.12, 57.375]),
        })

    if export_format == "torchscript":
        with torch.inference_mode():
            exported = torch.jit.trace(adapter, dummy, strict=False, check_trace=False)
            if bool(values.get("optimize", False)):
                exported = torch.jit.optimize_for_inference(exported)
            exported.save(str(output_path))
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif export_format == "onnx":
        if bool(values.get("int8", False)):
            raise NotImplementedError("ONNX INT8 quantization is not part of native graph export")
        _export_onnx(adapter, dummy, output_path, context, values)
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        if export_format == "engine" and context["device"].type != "cuda":
            raise ValueError("TensorRT export requires device=cuda")
        if export_format == "ncnn":
            with torch.inference_mode():
                torch.jit.trace(adapter, dummy, strict=False, check_trace=False).save(
                    str(output_dir / f"{stem}.torchscript")
                )
            torchscript_path = output_dir / f"{stem}.torchscript"
        else:
            torchscript_path = output_dir / f"{stem}.torchscript"
        with tempfile.TemporaryDirectory(prefix="animalpose-export-") as temporary:
            if export_format == "ncnn":
                torchscript_path = Path(temporary) / f"{stem}.torchscript"
                with torch.inference_mode():
                    torch.jit.trace(adapter, dummy, strict=False, check_trace=False).save(
                        str(torchscript_path)
                    )
            onnx_path = Path(temporary) / f"{stem}.onnx"
            if export_format not in {"coreml", "paddle", "ncnn"}:
                _export_onnx(adapter, dummy, onnx_path, context, values)
            result_path = export_with_backend(
                export_format,
                onnx_path=onnx_path,
                torchscript_path=torchscript_path,
                adapter=adapter,
                dummy=dummy,
                output_path=output_path,
                metadata=metadata,
                values=values,
                context=context,
            )
        output_path = Path(result_path)

    print(f"Export complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
