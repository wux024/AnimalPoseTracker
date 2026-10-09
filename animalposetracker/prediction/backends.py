"""Model artifact adapters for the project prediction workflow.

This module is intentionally independent from the GUI's live inference engine.  Every
backend receives the same BCHW float32 tensor contract and returns the model's raw pose
output; dataset-specific preprocessing and pose decoding stay in ``predict_cli``.
"""

import os

import platform

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from animalposetracker.artifacts import (
    EXPORT_FORMATS, PREDICT_FORMATS, detect_artifact_format, read_artifact_metadata,
)

from animalposetracker.artifacts import (
    EXPORT_FORMATS, PREDICT_FORMATS, detect_artifact_format, read_artifact_metadata,
)

def _ordered_outputs(outputs: Iterable[Any], names: Optional[List[str]] = None):
    if isinstance(outputs, dict):
        if names:
            ordered = [outputs[name] for name in names if name in outputs]
            if ordered:
                return ordered
        return list(outputs.values())
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]

class ModelArtifactBackend:
    """Load supported deployment artifacts without using the standalone engine."""

    def __init__(
        self,
        model_path,
        model_format: str,
        device,
        metadata: Optional[Dict[str, Any]] = None,
        use_opencv_dnn: bool = False,
    ) -> None:
        self.model_ref = str(model_path)
        self.model_format = str(model_format).lower()
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if self.model_format != "triton" else None
        )
        self.device = device
        if metadata is not None:
            self.metadata = metadata
        elif self.model_path is not None:
            self.metadata = read_artifact_metadata(self.model_path)
        else:
            self.metadata = {}
        self.output_names = list(self.metadata.get("outputs") or [])
        self._runner = self._load(use_opencv_dnn)

    def __call__(self, batch: np.ndarray):
        input_dtype = np.float16 if self.metadata.get("input_dtype") == "float16" else np.float32
        batch = np.ascontiguousarray(batch, dtype=input_dtype)
        return self._runner(batch)

    def _load(self, use_opencv_dnn: bool):
        model_format = self.model_format
        path = self.model_path
        if model_format == "torchscript":
            import torch

            model = torch.jit.load(str(path), map_location=self.device).eval()

            def run(batch):
                tensor = torch.from_numpy(batch).to(self.device)
                with torch.inference_mode():
                    output = model(tensor)
                return _ordered_outputs(output)

            return run

        if model_format in {"onnx", "imx"}:
            if model_format == "imx":
                candidate = next(path.glob("*.onnx"), None) if path.is_dir() else path
                if candidate is None:
                    raise FileNotFoundError(f"No ONNX graph was found in IMX model directory {path}")
                path = candidate
            if use_opencv_dnn:
                import cv2

                model = cv2.dnn.readNetFromONNX(str(path))

                def run(batch):
                    model.setInput(batch)
                    return _ordered_outputs(model.forward(model.getUnconnectedOutLayersNames()))

                return run
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError("ONNX project prediction requires `onnxruntime`") from exc
            providers = ["CPUExecutionProvider"]
            if getattr(self.device, "type", "cpu") == "cuda":
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.insert(0, "CUDAExecutionProvider")
            session_options = None
            if model_format == "imx":
                try:
                    import mct_quantizers

                    session_options = mct_quantizers.get_ort_session_options()
                except ImportError:
                    # Standard ONNX graphs can still run without the IMX quantizer helpers.
                    pass
            session = ort.InferenceSession(
                str(path), sess_options=session_options, providers=providers
            )
            input_name = session.get_inputs()[0].name

            def run(batch):
                outputs = session.run(None, {input_name: batch})
                return _ordered_outputs(outputs, self.output_names)

            return run

        if model_format == "openvino":
            try:
                from openvino import Core
            except ImportError as exc:
                raise ImportError("OpenVINO prediction requires the optional `openvino` package") from exc
            xml_path = path if path.suffix.lower() == ".xml" else next(path.glob("*.xml"), None)
            if xml_path is None:
                raise FileNotFoundError(f"No OpenVINO .xml graph was found at {path}")
            core = Core()
            compiled = core.compile_model(core.read_model(str(xml_path)), "AUTO")
            input_port = compiled.input(0)

            def run(batch):
                return _ordered_outputs(compiled({input_port: batch}), self.output_names)

            return run

        if model_format == "engine":
            return self._load_tensorrt()

        if model_format == "coreml":
            try:
                import coremltools as ct
            except ImportError as exc:
                raise ImportError("CoreML prediction requires the optional `coremltools` package") from exc
            model = ct.models.MLModel(str(path))
            input_name = model.get_spec().description.input[0].name

            def run(batch):
                return _ordered_outputs(model.predict({input_name: batch[0]}), self.output_names)

            return run

        if model_format in {"saved_model", "pb"}:
            return self._load_tensorflow(path)

        if model_format in {"tflite", "edgetpu"}:
            return self._load_tflite(path)

        if model_format == "paddle":
            return self._load_paddle(path)
        if model_format == "mnn":
            return self._load_mnn(path)
        if model_format == "ncnn":
            return self._load_ncnn(path)
        if model_format == "rknn":
            return self._load_rknn(path)
        if model_format == "tfjs":
            raise NotImplementedError(
                "TensorFlow.js is a browser deployment format and is not a local project-predict backend "
                "in the Ultralytics AutoBackend baseline. Use a TensorFlow SavedModel or TFLite artifact."
            )
        if model_format == "triton":
            return self._load_triton()
        raise ValueError(f"Unsupported project prediction format: {model_format}")

    def _load_triton(self):
        from urllib.parse import urlsplit

        reference = self.model_ref
        parsed = urlsplit(reference)
        if parsed.scheme not in {"http", "grpc"} or not parsed.netloc:
            raise ValueError(
                "Triton model references must look like http://host:8000/model or grpc://host:8001/model"
            )
        model_name = parsed.path.strip("/").split("/", 1)[0]
        if not model_name:
            raise ValueError("Triton URL must include the remote model name in its path")
        endpoint = parsed.netloc
        if parsed.scheme == "grpc":
            try:
                import tritonclient.grpc as triton
            except ImportError as exc:
                raise ImportError("Triton prediction requires `tritonclient[grpc]`") from exc
            client = triton.InferenceServerClient(url=endpoint)
        else:
            try:
                import tritonclient.http as triton
            except ImportError as exc:
                raise ImportError("Triton prediction requires `tritonclient[http]`") from exc
            client = triton.InferenceServerClient(url=endpoint)
        model_metadata = client.get_model_metadata(model_name)
        if isinstance(model_metadata, dict):
            input_description = model_metadata["inputs"][0]
            output_descriptions = model_metadata["outputs"]
            input_name = input_description["name"]
            input_datatype = input_description.get("datatype", "FP32")
            output_names = [item["name"] for item in output_descriptions]
        else:
            input_description = model_metadata.inputs[0]
            output_descriptions = model_metadata.outputs
            input_name = input_description.name
            input_datatype = input_description.datatype
            output_names = [item.name for item in output_descriptions]
        requested_outputs = [triton.InferRequestedOutput(name) for name in output_names]

        def run(batch):
            dtype = {
                "FP32": np.float32,
                "FP16": np.float16,
                "UINT8": np.uint8,
            }.get(input_datatype, np.float32)
            input_value = np.ascontiguousarray(batch, dtype=dtype)
            request = triton.InferInput(input_name, input_value.shape, input_datatype)
            request.set_data_from_numpy(input_value)
            response = client.infer(
                model_name=model_name,
                inputs=[request],
                outputs=requested_outputs,
            )
            return [response.as_numpy(name) for name in output_names]

        return run

    def _load_tensorrt(self):
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise ImportError("TensorRT prediction requires CUDA PyTorch and `tensorrt`") from exc
        if getattr(self.device, "type", "cpu") != "cuda":
            raise RuntimeError("TensorRT prediction requires a CUDA device")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.model_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine {self.model_path}")
        context = engine.create_execution_context()
        stream = torch.cuda.current_stream(self.device)

        if hasattr(engine, "num_io_tensors"):
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            input_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
            output_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

            def run(batch):
                tensor = torch.from_numpy(batch).to(self.device)
                input_name = input_names[0]
                context.set_input_shape(input_name, tuple(tensor.shape))
                context.set_tensor_address(input_name, int(tensor.data_ptr()))
                outputs = []
                for name in output_names:
                    shape = tuple(context.get_tensor_shape(name))
                    dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.float16 else torch.float32
                    output = torch.empty(shape, device=self.device, dtype=dtype)
                    context.set_tensor_address(name, int(output.data_ptr()))
                    outputs.append(output)
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT asynchronous inference failed")
                stream.synchronize()
                return [item.float().cpu().numpy() for item in outputs]

            return run

        input_indices = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
        output_indices = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
        binding_count = engine.num_bindings

        def run(batch):
            tensor = torch.from_numpy(batch).to(self.device)
            bindings = [0] * binding_count
            context.set_binding_shape(input_indices[0], tuple(tensor.shape))
            bindings[input_indices[0]] = int(tensor.data_ptr())
            outputs = []
            for index in output_indices:
                shape = tuple(context.get_binding_shape(index))
                dtype = torch.float16 if engine.get_binding_dtype(index) == trt.float16 else torch.float32
                output = torch.empty(shape, device=self.device, dtype=dtype)
                bindings[index] = int(output.data_ptr())
                outputs.append(output)
            if not context.execute_async_v2(bindings, stream.cuda_stream):
                raise RuntimeError("TensorRT asynchronous inference failed")
            stream.synchronize()
            return [item.float().cpu().numpy() for item in outputs]

        return run

    def _load_tensorflow(self, path: Path):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow prediction requires the optional `tensorflow` package") from exc
        if self.model_format == "saved_model":
            module = tf.saved_model.load(str(path))
            signature = module.signatures.get("serving_default")
            if signature is None:
                raise ValueError(f"TensorFlow SavedModel has no serving_default signature: {path}")
            input_name = next(iter(signature.structured_input_signature[1]))

            def run(batch):
                nhwc = np.transpose(batch, (0, 2, 3, 1))
                outputs = signature(**{input_name: tf.convert_to_tensor(nhwc)})
                return _ordered_outputs({key: value.numpy() for key, value in outputs.items()}, self.output_names)

            return run

        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(path.read_bytes())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
        session = tf.compat.v1.Session(graph=graph)
        input_name = self.metadata.get("input_tensor", "images:0")
        output_names = self.metadata.get("output_tensors") or [
            f"{name}:0" for name in self.output_names
        ]
        if not output_names or any(not name for name in output_names):
            raise ValueError("TensorFlow GraphDef artifact metadata must name output_tensors")
        input_tensor = graph.get_tensor_by_name(input_name)
        output_tensors = [graph.get_tensor_by_name(name) for name in output_names]

        def run(batch):
            nhwc = np.transpose(batch, (0, 2, 3, 1))
            return session.run(output_tensors, feed_dict={input_tensor: nhwc})

        return run

    def _load_tflite(self, path: Path):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise ImportError("TFLite prediction requires `tflite-runtime` or `tensorflow`") from exc
            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate
        kwargs = {"model_path": str(path)}
        if self.model_format == "edgetpu":
            try:
                delegate = {
                    "Linux": "libedgetpu.so.1",
                    "Darwin": "libedgetpu.1.dylib",
                    "Windows": "edgetpu.dll",
                }[platform.system()]
                kwargs["experimental_delegates"] = [load_delegate(delegate)]
            except (OSError, ValueError) as exc:
                raise RuntimeError("Edge TPU prediction requires the Coral Edge TPU runtime") from exc
        interpreter = Interpreter(**kwargs)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()

        def run(batch):
            value = np.transpose(batch, (0, 2, 3, 1))
            dtype = input_detail["dtype"]
            if np.issubdtype(dtype, np.integer):
                scale, zero_point = input_detail["quantization"]
                if scale:
                    value = np.rint(value / scale + zero_point)
                value = np.clip(value, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
            else:
                value = value.astype(dtype)
            interpreter.set_tensor(input_detail["index"], value)
            interpreter.invoke()
            outputs = []
            for detail in output_details:
                result = interpreter.get_tensor(detail["index"])
                scale, zero_point = detail["quantization"]
                if np.issubdtype(result.dtype, np.integer) and scale:
                    result = (result.astype(np.float32) - zero_point) * scale
                outputs.append(result)
            return _ordered_outputs(outputs, self.output_names)

        return run

    def _load_paddle(self, path: Path):
        try:
            import paddle.inference as paddle_infer
        except ImportError as exc:
            raise ImportError("PaddlePaddle prediction requires the optional `paddlepaddle` package") from exc
        model_file = path if path.suffix.lower() in {".pdmodel", ".json"} else next(
            iter(list(path.glob("*.pdmodel")) + list(path.glob("*.json"))), None
        )
        if model_file is None:
            raise FileNotFoundError(f"No Paddle model graph (.pdmodel or .json) was found at {path}")
        params_file = model_file.with_suffix(".pdiparams")
        config = paddle_infer.Config(str(model_file), str(params_file))
        config.disable_glog_info()
        config.disable_gpu()
        predictor = paddle_infer.create_predictor(config)
        input_name = predictor.get_input_names()[0]
        output_names = predictor.get_output_names()

        def run(batch):
            handle = predictor.get_input_handle(input_name)
            handle.reshape(batch.shape)
            handle.copy_from_cpu(batch)
            predictor.run()
            outputs = [predictor.get_output_handle(name).copy_to_cpu() for name in output_names]
            return _ordered_outputs(outputs, self.output_names)

        return run

    def _load_mnn(self, path: Path):
        try:
            import MNN
        except ImportError as exc:
            raise ImportError("MNN prediction requires the optional `MNN` Python package") from exc
        manager = MNN.nn.create_runtime_manager(({
            "precision": "low",
            "backend": "CPU",
            "numThread": max(1, (os.cpu_count() or 1) // 2),
        },))
        module = MNN.nn.load_module_from_file(
            str(path), [], [], runtime_manager=manager, rearrange=True
        )

        def run(batch):
            input_var = MNN.expr.const(batch.ctypes.data, batch.shape)
            outputs = module.onForward([input_var])
            return [np.asarray(output.read()) for output in outputs]

        return run

    def _load_ncnn(self, path: Path):
        try:
            import ncnn
        except ImportError as exc:
            raise ImportError("NCNN prediction requires the optional `ncnn` Python package") from exc
        param_file = path if path.suffix.lower() == ".param" else next(path.glob("*.param"), None)
        if param_file is None:
            raise FileNotFoundError(f"No NCNN .param file was found at {path}")
        net = ncnn.Net()
        net.load_param(str(param_file))
        net.load_model(str(param_file.with_suffix(".bin")))
        input_names = list(net.input_names())
        available_outputs = sorted(net.output_names())
        input_name = self.metadata.get("input_name") or input_names[0]
        output_names = [name for name in self.output_names if name in available_outputs]
        output_names = output_names or available_outputs

        def run(batch):
            outputs = []
            for sample in batch:
                extractor = net.create_extractor()
                extractor.input(input_name, ncnn.Mat(np.ascontiguousarray(sample)))
                for name in output_names:
                    status, result = extractor.extract(name)
                    if status != 0:
                        raise RuntimeError(f"NCNN failed to extract output {name!r}: status {status}")
                    outputs.append(np.asarray(result))
            return outputs

        return run

    def _load_rknn(self, path: Path):
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise ImportError("RKNN prediction requires the optional `rknn-toolkit-lite2` package") from exc
        model_file = path if path.suffix.lower() == ".rknn" else next(path.rglob("*.rknn"), None)
        if model_file is None:
            raise FileNotFoundError(f"No RKNN model file was found at {path}")
        runtime = RKNNLite()
        if runtime.load_rknn(str(model_file)) != 0:
            raise RuntimeError(f"RKNN failed to load {model_file}")
        if runtime.init_runtime() != 0:
            raise RuntimeError("RKNN failed to initialize its device runtime")

        def run(batch):
            outputs = []
            for sample in batch:
                sample = sample.copy()
                if "pixel_mean" in self.metadata and "pixel_std" in self.metadata:
                    mean = np.asarray(self.metadata["pixel_mean"], dtype=np.float32)[:, None, None]
                    std = np.asarray(self.metadata["pixel_std"], dtype=np.float32)[:, None, None]
                    sample = sample * std + mean
                else:
                    sample = sample * 255.0
                nhwc = np.transpose(sample, (1, 2, 0))[None].clip(0, 255).astype(np.uint8)
                result = runtime.inference(inputs=[nhwc])
                if result is None:
                    raise RuntimeError("RKNN inference returned no outputs")
                outputs.extend(result)
            return _ordered_outputs(outputs, self.output_names)

        return run

__all__ = [
    "EXPORT_FORMATS", "PREDICT_FORMATS", "ModelArtifactBackend",
    "detect_artifact_format", "read_artifact_metadata",
]
