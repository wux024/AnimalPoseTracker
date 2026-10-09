"""Optional deployment-format converters for native AnimalPoseTracker models."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _require(module_name: str, pip_name: str = None):
    try:
        return __import__(module_name.split(">", 1)[0].split("=", 1)[0])
    except ImportError as exc:
        raise ImportError(
            f"This export target requires `{pip_name or module_name}` in the active environment"
        ) from exc


def _save_metadata(path: Path, metadata: Dict[str, Any]):
    path = Path(path)
    if path.is_dir() and path.suffix.lower() != ".mlpackage":
        sidecar = path / "metadata.json"
    else:
        sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _export_openvino(onnx_path: Path, output_path: Path, metadata, half=False):
    ov = _require("openvino")
    output_path.mkdir(parents=True, exist_ok=True)
    xml_path = output_path / f"{onnx_path.stem}.xml"
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(half))
    _save_metadata(output_path, metadata)
    return output_path


def _export_tensorrt(onnx_path: Path, output_path: Path, metadata, values, dummy_shape):
    trt = _require("tensorrt")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the exported ONNX graph:\n{errors}")

    config = builder.create_builder_config()
    workspace = values.get("workspace")
    workspace_bytes = int(float(workspace or 4.0) * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if bool(values.get("half", False)) and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if bool(values.get("dynamic", False)):
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        h, w = int(dummy_shape[-2]), int(dummy_shape[-1])
        input_name = input_tensor.name
        profile.set_shape(input_name, (1, 3, max(32, h // 2), max(32, w // 2)),
                          tuple(dummy_shape), (max(2, int(dummy_shape[0]) * 4), 3, h * 2, w * 2))
        config.add_optimization_profile(profile)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized = engine.serialize() if engine is not None else None
    if serialized is None:
        raise RuntimeError("TensorRT failed to build an engine for this pose model")
    output_path.write_bytes(bytes(serialized))
    _save_metadata(output_path, metadata)
    return output_path


def _export_coreml(adapter, dummy, output_path: Path, metadata, half=False):
    ct = _require("coremltools>=8.0", "coremltools>=8.0")
    import torch

    if dummy.device.type != "cpu":
        adapter = adapter.to("cpu").eval()
        dummy = dummy.cpu()
    traced = torch.jit.trace(adapter, dummy, strict=False, check_trace=False)
    inputs = [ct.TensorType(name="images", shape=tuple(dummy.shape))]
    kwargs = {"inputs": inputs}
    if half and hasattr(ct, "precision"):
        kwargs["compute_precision"] = ct.precision.FLOAT16
    converted = ct.convert(traced, **kwargs)
    converted.save(str(output_path))
    _save_metadata(output_path, metadata)
    return output_path


def _onnx_to_tensorflow(onnx_path: Path, output_dir: Path):
    _require("tensorflow>=2.0.0", "tensorflow>=2.0.0")
    onnx2tf = _require("onnx2tf>=1.26.3", "onnx2tf>=1.26.3")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        not_use_onnxsim=True,
        verbosity="error",
        output_signaturedefs=True,
        enable_batchmatmul_unfold=True,
    )
    saved_model = output_dir if (output_dir / "saved_model.pb").is_file() else next(
        (path for path in output_dir.rglob("saved_model.pb")), None
    )
    if saved_model is None:
        raise RuntimeError("onnx2tf completed without producing saved_model.pb")
    return saved_model.parent


def _representative_dataset(context, metadata, values, sample_count=100):
    """Yield model-ready NHWC calibration samples from the configured val/test split."""
    config_path = context.get("dataset_config_path")
    if config_path is None:
        raise ValueError("INT8 calibration requires the project dataset.yaml")
    from animalposetracker.data.pose import PoseTextDataset

    data_values = context["dataset"]
    split = "val" if data_values.get("val") else "test"
    width, height = map(int, metadata["input_size"][:2])
    if context.get("head_name") == "SimCCHead":
        from animalposetracker.data.topdown import TopDownPoseDataset

        dataset = TopDownPoseDataset(
            config_path,
            split,
            input_size=(width, height),
            sigma=float(context.get("other", {}).get("label_sigma", 6.0)),
            split_ratio=float(metadata.get("simcc_split_ratio", 2.0)),
            preprocessing_config=context.get("model_spec", {}).get("preprocessing") or {},
        )
        total = min(len(dataset), int(sample_count))
        if total < 1:
            raise ValueError("The calibration split contains no top-down pose instances")
        for index in range(total):
            chw = dataset[index]["images"].numpy()
            yield [np.transpose(chw[None], (0, 2, 3, 1)).astype(np.float32)]
        return

    dataset = PoseTextDataset(config_path, split=split, image_size=max(width, height), cache=False)
    image_paths = dataset.image_paths[: int(sample_count)]
    if not image_paths:
        raise ValueError("The calibration split contains no images")
    for image_path in image_paths:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        source_height, source_width = image.shape[:2]
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        left, top = (width - resized_width) // 2, (height - resized_height) // 2
        canvas[top:top + resized_height, left:left + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        yield [rgb[None]]


def _export_tensorflow(onnx_path: Path, output_path: Path, export_format: str, metadata, values, context):
    import tensorflow as tf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with __import__("tempfile").TemporaryDirectory(prefix="animalpose-tf-") as temporary:
        saved_model_dir = _onnx_to_tensorflow(onnx_path, Path(temporary) / "saved_model")
        if export_format == "saved_model":
            if output_path.exists():
                shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()
            shutil.copytree(saved_model_dir, output_path)
            _save_metadata(output_path, metadata)
            return output_path

        module = tf.saved_model.load(str(saved_model_dir))
        signature = module.signatures.get("serving_default")
        if signature is None:
            raise RuntimeError("Converted TensorFlow model has no serving_default signature")
        if export_format == "pb":
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

            frozen = convert_variables_to_constants_v2(signature)
            tf.io.write_graph(frozen.graph, str(output_path.parent), output_path.name, as_text=False)
            _save_metadata(output_path, {
                **metadata,
                "input_tensor": frozen.inputs[0].name,
                "output_tensors": [tensor.name for tensor in frozen.outputs],
            })
            return output_path

        if export_format in {"tflite", "edgetpu"}:
            int8 = bool(values.get("int8", False)) or export_format == "edgetpu"
            if export_format == "edgetpu" and bool(values.get("half", False)):
                raise ValueError("Edge TPU does not accept FP16 export")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            if int8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = lambda: _representative_dataset(
                    context, metadata, values
                )
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            elif bool(values.get("half", False)):
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            converted = converter.convert()
            if export_format == "tflite":
                output_path.write_bytes(converted)
                _save_metadata(output_path, metadata)
                return output_path

            if not shutil.which("edgetpu_compiler"):
                raise FileNotFoundError(
                    "Edge TPU export requires the platform-specific `edgetpu_compiler` executable"
                )
            with tempfile.TemporaryDirectory(prefix="animalpose-edgetpu-") as temporary:
                tflite_path = Path(temporary) / "calibrated.tflite"
                tflite_path.write_bytes(converted)
                subprocess.run(
                    [shutil.which("edgetpu_compiler"), "--out_dir", temporary, str(tflite_path)],
                    check=True,
                )
                compiled = next(Path(temporary).glob("*_edgetpu.tflite"), None)
                if compiled is None:
                    raise RuntimeError("Edge TPU compiler did not produce an _edgetpu.tflite artifact")
                shutil.copy2(compiled, output_path)
            _save_metadata(output_path, metadata)
            return output_path

        if export_format == "edgetpu":
            raise AssertionError("Edge TPU export is handled alongside TFLite conversion")

        if export_format == "tfjs":
            converter_path = shutil.which("tensorflowjs_converter")
            if converter_path is None:
                raise FileNotFoundError("TensorFlow.js export requires `tensorflowjs_converter` on PATH")
            command = [converter_path, "--input_format=tf_saved_model"]
            if bool(values.get("half", False)):
                command.append("--quantize_float16")
            command.extend((str(saved_model_dir), str(output_path)))
            subprocess.run(command, check=True)
            _save_metadata(output_path, metadata)
            return output_path
    raise ValueError(f"Unsupported TensorFlow export target {export_format!r}")


def _export_paddle(adapter, dummy, output_path: Path, metadata):
    _require("paddle", "paddlepaddle")
    _require("x2paddle")
    from x2paddle.convert import pytorch2paddle

    pytorch2paddle(module=adapter.eval().cpu(), save_dir=str(output_path), jit_type="trace", input_examples=[dummy.cpu()])
    _save_metadata(output_path, metadata)
    return output_path


def _export_mnn(onnx_path: Path, output_path: Path, metadata, values):
    _require("MNN>=2.9.6", "MNN>=2.9.6")
    from MNN.tools import mnnconvert

    arguments = [
        "", "-f", "ONNX", "--modelFile", str(onnx_path), "--MNNModel", str(output_path),
        "--bizCode", json.dumps(metadata),
    ]
    if bool(values.get("int8", False)):
        arguments.extend(("--weightQuantBits", "8"))
    if bool(values.get("half", False)):
        arguments.append("--fp16")
    mnnconvert.convert(arguments)
    _save_metadata(output_path, metadata)
    return output_path


def _export_ncnn(torchscript_path: Path, output_path: Path, metadata, dummy, values):
    if not shutil.which("pnnx"):
        raise FileNotFoundError("NCNN export requires the PNNX converter executable on PATH")
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = output_path / "model"
    command = [
        shutil.which("pnnx"), str(torchscript_path),
        f"ncnnparam={prefix}.ncnn.param",
        f"ncnnbin={prefix}.ncnn.bin",
        f"inputshape={[int(dummy.shape[0]), 3, int(dummy.shape[-2]), int(dummy.shape[-1])]}" ,
        f"fp16={int(bool(values.get('half', False)))}",
    ]
    subprocess.run(command, check=True)
    _require("ncnn")
    import ncnn

    net = ncnn.Net()
    net.load_param(str(prefix) + ".ncnn.param")
    metadata = dict(metadata)
    metadata["input_name"] = list(net.input_names())[0]
    metadata["outputs"] = sorted(net.output_names())
    _save_metadata(output_path, metadata)
    return output_path


def _export_rknn(onnx_path: Path, output_path: Path, metadata, values):
    rknn_module = _require("rknn", "rknn-toolkit2")
    RKNN = getattr(rknn_module, "RKNN", None)
    if RKNN is None:
        from rknn.api import RKNN
    output_path.mkdir(parents=True, exist_ok=True)
    target = str(values.get("rknn_target") or "rk3588")
    converter = RKNN(verbose=False)
    try:
        if "pixel_mean" in metadata and "pixel_std" in metadata:
            mean_values = [metadata["pixel_mean"]]
            std_values = [metadata["pixel_std"]]
        else:
            mean_values = [[0, 0, 0]]
            std_values = [[255, 255, 255]]
        converter.config(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target,
        )
        if converter.load_onnx(model=str(onnx_path)) != 0:
            raise RuntimeError("RKNN failed to load the ONNX graph")
        if converter.build(do_quantization=False) != 0:
            raise RuntimeError("RKNN failed to compile the ONNX graph")
        artifact_path = output_path / f"{onnx_path.stem}-{target}.rknn"
        if converter.export_rknn(str(artifact_path)) != 0:
            raise RuntimeError("RKNN failed to write the compiled artifact")
    finally:
        converter.release()
    _save_metadata(output_path, metadata)
    return output_path


def _export_imx(adapter, dummy, output_path: Path, metadata, context):
    """Build a Sony IMX500 INT8 graph with its target platform calibration toolkit."""
    if not str(__import__("sys").platform).startswith("linux"):
        raise RuntimeError("Sony IMX500 export is supported on Linux only")
    mct = _require("model_compression_toolkit", "model-compression-toolkit>=2.4.1")
    tpc_module = _require("edgemdt_tpc", "edge-mdt-tpc>=1.1.0")
    onnx = _require("onnx>=1.12.0", "onnx>=1.12.0")
    import torch

    compiler = shutil.which("imxconv-pt")
    if compiler is None:
        raise FileNotFoundError(
            "Sony IMX500 export requires the vendor `imx500-converter[pt]` package and imxconv-pt executable"
        )
    if "dataset_config_path" not in context:
        raise ValueError("IMX export requires the project dataset for PTQ calibration")

    def representative_dataset_gen():
        for [nhwc] in _representative_dataset(context, metadata, {}, sample_count=100):
            nchw = np.transpose(nhwc, (0, 3, 1, 2)).copy()
            yield [torch.from_numpy(nchw).to(dtype=dummy.dtype)]

    tpc = tpc_module.get_target_platform_capabilities(tpc_version="4.0", device_type="imx500")
    quant_model = mct.ptq.pytorch_post_training_quantization(
        in_module=adapter.eval().cpu(),
        representative_data_gen=representative_dataset_gen,
        core_config=mct.core.CoreConfig(),
        target_platform_capabilities=tpc,
    )[0]
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / "pose_imx.onnx"
    mct.exporter.pytorch_export_model(
        model=quant_model,
        save_model_path=onnx_path,
        repr_dataset=representative_dataset_gen,
    )
    model_onnx = onnx.load(str(onnx_path))
    for key, value in metadata.items():
        property_value = model_onnx.metadata_props.add()
        property_value.key = str(key)
        property_value.value = json.dumps(value, ensure_ascii=False)
    onnx.save(model_onnx, str(onnx_path))
    subprocess.run(
        [compiler, "-i", str(onnx_path), "-o", str(output_path), "--no-input-persistency", "--overwrite-output"],
        check=True,
    )
    labels = context["dataset"].get("names") or {}
    if isinstance(labels, dict):
        labels = [labels[key] for key in sorted(labels, key=lambda item: int(item))]
    (output_path / "labels.txt").write_text("".join(f"{label}\n" for label in labels), encoding="utf-8")
    _save_metadata(output_path, metadata)
    return output_path


def export_with_backend(
    export_format: str,
    *,
    onnx_path: Path,
    torchscript_path: Path,
    adapter,
    dummy,
    output_path: Path,
    metadata: Dict[str, Any],
    values: Dict[str, Any],
    context: Dict[str, Any],
):
    """Export a portable native graph through an optional platform-specific converter."""
    if export_format == "openvino":
        return _export_openvino(onnx_path, output_path, metadata, values.get("half", False))
    if export_format == "engine":
        return _export_tensorrt(onnx_path, output_path, metadata, values, dummy.shape)
    if export_format == "coreml":
        return _export_coreml(adapter, dummy, output_path, metadata, values.get("half", False))
    if export_format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
        return _export_tensorflow(onnx_path, output_path, export_format, metadata, values, context)
    if export_format == "paddle":
        return _export_paddle(adapter, dummy, output_path, metadata)
    if export_format == "mnn":
        return _export_mnn(onnx_path, output_path, metadata, values)
    if export_format == "ncnn":
        return _export_ncnn(torchscript_path, output_path, metadata, dummy, values)
    if export_format == "rknn":
        return _export_rknn(onnx_path, output_path, metadata, values)
    if export_format == "imx":
        return _export_imx(adapter, dummy, output_path, metadata, context)
    raise ValueError(f"No deployment converter is registered for format {export_format!r}")


__all__ = ["export_with_backend"]
