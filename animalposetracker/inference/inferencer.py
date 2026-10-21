from typing import Dict, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
import operator
import yaml
from pathlib import Path
import os

from .constant import ENGINEtoBackend, OpenCV_TARGETS, EP_PARAMS


def _static_shape(shape) -> Tuple[int, ...]:
    """Normalise a backend-reported tensor shape to plain ints.

    Only static, non-negative dimensions survive; anything else (ONNX Runtime's
    symbolic dimension names, OpenVINO's dynamic markers, ``None``, negative
    sentinels) becomes ``-1``. Consumers treat ``-1`` as "unknown, skip me" --
    see ``inference.cli._resolve_num_classes`` and
    ``inference.schema_resolution.resolve_output_schema`` -- so the sentinel has
    to be uniform across engines rather than each backend's own convention.

    Dimensions are converted through ``operator.index`` rather than ``int()`` so
    that integer-like scalars a backend may hand back (``numpy.int64`` from a
    descriptor, for instance) count as static. ``bool`` is rejected explicitly
    because it is integer-like but is not a dimension.
    """
    normalised = []
    for dimension in shape:
        try:
            value = operator.index(dimension) if not isinstance(dimension, bool) else None
        except TypeError:
            value = None
        normalised.append(value if value is not None and value >= 0 else -1)
    return tuple(normalised)


def _partial_shape_to_static(partial) -> Tuple[int, ...]:
    """Convert an OpenVINO partial shape into static ints, ``-1`` per dynamic dim.

    Duck-typed on purpose: the caller passes whatever ``Output.get_partial_shape()``
    returned, and this module never imports OpenVINO at import time. Returns an
    empty tuple when the rank itself is dynamic, which callers read as "unknown".
    """
    try:
        if not partial.rank.is_static:
            return ()
        return tuple(
            int(dimension.get_length()) if dimension.is_static else -1
            for dimension in partial
        )
    except Exception:
        return ()


class InferenceEngine:
    def __init__(self,
                 config: Union[str, Path, Dict] = None,
                 weights_path: Union[str, Path] = None,
                 input_width: int = 640,
                 input_height: int = 640,
                 engine: str = 'ONNX',
                 device: str = 'Intel CPU',
                 model_bits: str = 'FP32',
                 conf: float = 0.25,
                 iou: float = 0.45,
                 background: bool = 'Original',
                 show_classes: bool = False,
                 show_keypoints: bool = True,
                 show_skeletons: bool = False,
                 show_bbox: bool = False,
                 radius: int = 5,
                 skeleton_line_width: int = 2,
                 bbox_line_width: int = 2,
                 show_fps: bool = True,
                 show_preprocess_time: bool = True,
                 show_inference_time: bool = True,
                 show_postprocess_time: bool = True,
                 font_scale: float = 1.0,
                 output_names: Optional[Sequence[str]] = None,
                 ):
        self.model = None
        self._weights_path = weights_path
        self._engine = engine
        self._device = device
        self._model_bits = model_bits
        self._coreml = False
        self._tensorrt = False
        self._input_width = input_width
        self._input_height = input_height
        self._configured_output_names = tuple(output_names or ())
        self._runtime_output_names = ()
        self._runtime_output_shapes = {}
        self.visualize_config = {
            'conf': conf,
            'iou': iou,
           'show_classes': show_classes,
           'show_keypoints': show_keypoints,
           'show_skeletons': show_skeletons,
           'show_bbox': show_bbox,
            'radius': radius,
           'skeleton_line_width': skeleton_line_width,
            'bbox_line_width': bbox_line_width,
            'background': background,
            'show_fps': show_fps,
           'show_preprocess_time': show_preprocess_time,
           'show_inference_time': show_inference_time,
           'show_postprocess_time': show_postprocess_time,
            'font_scale': font_scale,
        }
        self._data_config = None
        self._load_config(config)

    @property
    def data_config(self):
        return self._data_config

    @data_config.setter
    def data_config(self, config: Union[str, Path, Dict]):
        self._data_config = config
        self._update_config_vars()
        
    
    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
    
    @property
    def model_bits(self):
        return self._model_bits

    @model_bits.setter
    def model_bits(self, model_bits):
        self._model_bits = model_bits
    
    @property
    def input_width(self):
        return self._input_width

    @input_width.setter
    def input_width(self, input_width):
        self._input_width = input_width

    @property
    def input_height(self):
        return self._input_height

    @input_height.setter
    def input_height(self, input_height):
        self._input_height = input_height

    @property
    def weights_path(self):
        return self._weights_path

    @weights_path.setter
    def weights_path(self, weights_path):
        self._weights_path = weights_path

    @property
    def output_names(self):
        """Names used to request model outputs, when the backend exposes them."""
        return self._configured_output_names or self._runtime_output_names

    def print_config(self):
        print(f"Engine: {self._engine}")
        print(f"Device: {self._device}")
        print(f"Model Bits: {self._model_bits}")
        print(f"Input Width: {self._input_width}")
        print(f"Input Height: {self._input_height}")
        for key, value in self.visualize_config.items():
            print(f"{key}: {value}")


    def _update_config_vars(self):
        if self._data_config is None:
            return
        
        if isinstance(self._data_config, (str, Path)):
            # Load the data config file
            with open(self._data_config, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(self._data_config, dict):
            config = self._data_config
        else:
            raise ValueError("Invalid data config type. Please provide either a path to a YAML file or a dictionary.")

        self.classes = config.get('classes_name', [])
        self.keypoints = config.get('keypoint_names') or config.get('keypoints_name', [])
        self.skeleton = config.get('skeleton', [])

        # Generate a color palette for the classes
        self.classes_color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.keypoints_color_palette = np.random.uniform(0, 255, size=(len(self.keypoints), 3))
        self.skeleton_color_palette = np.random.uniform(0, 255, size=(len(self.skeleton), 3))

    def update_config(self, config: Dict):
        for key, value in config.items():
            if key in self.visualize_config:
                self.visualize_config[key] = value
            else:
                raise ValueError(f"Invalid key {key} in config. Please choose from {self.visualize_config.keys()}")

    def _load_config(self, config: Union[str, Path, Dict] = None):
        self.data_config = config

    def model_init(self):
        self._runtime_output_names = ()
        self._runtime_output_shapes = {}
        engine_init_method = {
            'OpenCV': self._init_opencv,
            'ONNX': self._init_onnx,
            'OpenVINO': self._init_openvino,
            'TensorRT': self._init_tensorrt,
            'CoreML': self._init_coreml,
            'CANN': self._init_cann,

        }
        if self._engine == "CoreML":
            self._coreml = True
        elif self._engine == "TensorRT":
            self._tensorrt = True
        else:
            self._coreml = False
            self._tensorrt = False
        self.print_config()
        engine_init_method[self._engine]()
    
    def _init_model_input_shape(self):
        weights_path = Path(self._weights_path)
        if weights_path.suffix == '.onnx':
            try:
                import onnxruntime as ort
                model = ort.InferenceSession(self._weights_path)
                input_shape = model.get_inputs()[0].shape
                if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                    self._input_width = input_shape[2]
                    self._input_height = input_shape[3]
            except ImportError:
                raise ImportError("Please install onnxruntime to use ONNX engine.")
        elif (weights_path.suffix == '.xml' and
              weights_path.with_suffix('.bin').exists() and
              self._device in ["Intel GPU", "Intel NPU"]):
            inputs = self.model.inputs
            for input_layer in inputs:
                input_shape = input_layer.shape
                if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                    self._input_width = input_shape[2]
                    self._input_height = input_shape[3]
        elif weights_path.suffix == '.om':
            pass
        else:
            raise ValueError("Invalid weights file format. "
                             "Please provide either an ONNX or an XML+BIN file.")
            
    def _init_opencv(self):
        cv2_backends = ENGINEtoBackend["OpenCV"]
        cv2_backend, cv2_target = self._get_backend_and_target(cv2_backends)
        self.model = self._load_model()
        self._init_model_input_shape()
        self._runtime_output_names = tuple(
            str(name) for name in self.model.getUnconnectedOutLayersNames()
        )
        self._runtime_output_shapes = self._probe_opencv_output_shapes()
        self.model.setPreferableBackend(cv2_backend)
        self.model.setPreferableTarget(cv2_target)

    def _probe_opencv_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Ask the OpenCV DNN graph for its output shapes before any inference.

        ``getLayersShapes`` runs a full shape-inference pass over the network,
        which is not free, and it is not available on every build/target. It is
        therefore best-effort: on any failure the mapping stays empty and the
        shapes are backfilled from the first real forward pass instead (see
        ``inference_named_outputs``). An empty mapping is a normal outcome, not
        an error -- callers that need shapes before the first frame must supply
        the schema through configuration.
        """
        try:
            input_shape = (1, 3, int(self._input_height), int(self._input_width))
            _, _, layer_shapes = self.model.getLayersShapes(input_shape)
            probed: Dict[str, Tuple[int, ...]] = {}
            for name in self._runtime_output_names:
                layer_id = int(self.model.getLayerId(name))
                if layer_id < 0:
                    continue
                outputs = layer_shapes[layer_id]
                if not outputs:
                    continue
                probed[name] = _static_shape(outputs[0])
            return probed
        except Exception:
            # Shape probing is optional; the decoder validates shapes itself, so
            # a missing probe only costs the automatic schema detection path.
            return {}

    def _get_backend_and_target(self, cv2_backends):
        """
        Get the appropriate OpenCV backend based on the device.
        :param cv2_backends: Mapping of device types to OpenCV backends
        :return: OpenCV backend
        """
        try:    
            if "CPU" in self._device:
                cv2_backend = cv2_backends.get("CPU")
            else:
                cv2_backend = cv2_backends.get(self._device)
        except KeyError:
            raise ValueError(f"Invalid device {self._device}. Please choose from {cv2_backends.keys()}")
        
        if len(cv2.dnn.getAvailableTargets(cv2_backend)) == 0:
            if "Intel GPU" in self._device:
                print("You could build opencv from source with OpenVINO support to "
                      "use Intel GPU as device. We will use default OpenCV backend instead.")
                cv2_backend = cv2.dnn.DNN_BACKEND_OPENCV
                config_path = Path.cwd() / "OCL4DNN_CONFIG_CACHE"
                config_path.mkdir(exist_ok=True)
                os.environ['OPENCV_OCL4DNN_CONFIG_PATH'] = str(config_path)
                if self._model_bits == "FP16":
                    cv2_target = cv2.dnn.DNN_TARGET_OPENCL_FP16
                else:
                    cv2_target = cv2.dnn.DNN_TARGET_OPENCL
                return cv2_backend, cv2_target
            elif "NVIDIA GPU" in self._device:
                raise ValueError("You could build opencv from source with CUDA support to "
                                 "use NVIDIA GPU as device.")
            elif "Intel NPU" in self._device:
                raise ValueError("You could build opencv from source with OpenVINO support to "
                                 "use Intel NPU as device.")
            elif "Ascend NPU" in self._device:
                raise ValueError("You could build opencv from source with CANN support to "
                                 "use Ascend NPU as device.")
            elif "Metal" in self._device:
                raise ValueError("You could build opencv from source with Metal support to "
                                 "use Metal as device.")

            target_key = f"{self._device} FP16"
        else:
            target_key = self._device
        return cv2_backend, OpenCV_TARGETS.get(target_key)

    def _load_model(self):
        """
        Load the model based on the weights file format.
        :return: Loaded OpenCV model
        """
        if Path(self._weights_path).suffix == '.onnx':
            return cv2.dnn.readNetFromONNX(self._weights_path)
        elif (Path(self._weights_path).suffix == '.xml' and
              Path(self._weights_path).with_suffix('.bin').exists() and
              self._device in ["Intel GPU", "Intel NPU"]):
            xml_path = Path(self._weights_path)
            bin_path = Path(self._weights_path).with_suffix('.bin')
            return cv2.dnn.readNetFromModelOptimizer(str(xml_path), str(bin_path))
        elif Path(self._weights_path).suffix == '.om':
            return cv2.dnn.readNet(self._weights_path)
        else:
            raise ValueError("Invalid weights file format. "
                             "Please provide either an ONNX or an XML+BIN file.")

    def _init_onnx(self, params=False):
        """Initialize ONNX Runtime session with optimal execution providers and parameters."""
        try:
            import onnxruntime as ort
            
            # Get the recommended providers for the current device
            providers_list = ENGINEtoBackend["ONNX"][self._device]
            providers_available = ort.get_available_providers()
            providers_used = []
            

            # Build the final provider list with parameters
            for provider in providers_list:
                if provider in providers_available:
                    if params:
                        providers_used.append(self.get_provider_config(provider))
                    else:
                        providers_used.append(provider)
                        
            print(f"Initializing ONNX with providers: {providers_used}")
            
            # Session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Initialize the inference session
            self.model = ort.InferenceSession(
                self._weights_path,
                sess_options=sess_options,
                providers=providers_used
            )
            
            # Get input/output names
            self.input_name = self.model.get_inputs()[0].name
            outputs = self.model.get_outputs()
            self._runtime_output_names = tuple(output.name for output in outputs)
            self.output_name = self._runtime_output_names[0]
            # ONNX Runtime reports a symbolic dimension as a str, which
            # _static_shape turns into -1; the schema resolver then skips it.
            self._runtime_output_shapes = {
                output.name: _static_shape(output.shape) for output in outputs
            }

            input_shape = self.model.get_inputs()[0].shape

            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                self._input_width = input_shape[2]
                self._input_height = input_shape[3]
            
            # Print active provider for verification
            print(f"ONNX session successfully initialized with provider: {self.model.get_providers()[0]}")
            
        except ImportError:
            raise ImportError("Please install onnxruntime to use ONNX engine.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX Runtime session: {str(e)}")
    
    def get_provider_config(self, provider_name):
        """Get properly configured provider parameters based on model bits and device"""
        if provider_name not in EP_PARAMS:
            return provider_name  # No special parameters
        
        # Get default parameters for this provider
        params = {}
        for param, config in EP_PARAMS[provider_name].items():
            params[param] = config["default"]
        
        # Device-specific configurations
        device_id = getattr(self, 'device_id', 0)
        model_bits = getattr(self, 'model_bits', 'FP32')
        
        # Configure provider-specific parameters
        if provider_name == "CUDAExecutionProvider":
            params.update({
                "device_id": device_id,
                "cudnn_conv_algo_search": 1 if model_bits == 'FP16' else 0
            })
        
        elif provider_name == "TensorrtExecutionProvider":
            params.update({
                "device_id": device_id,
                "trt_fp16_enable": model_bits == 'FP16',
                "trt_int8_enable": model_bits == 'INT8',
                "trt_engine_cache_enable": True
            })
        
        elif provider_name == "OpenVINOExecutionProvider":
            device_type = "CPU"
            if "GPU" in self._device:
                device_type = f"GPU.{device_id}" if device_id > 0 else "GPU"
            elif "NPU" in self._device:
                device_type = "NPU"
            
            params.update({
                "device_type": device_type,
                "num_of_threads": getattr(self, 'num_threads', 8),
                "precision": model_bits
            })
        
        elif provider_name == "DmlExecutionProvider":
            params["device_id"] = device_id
        
        elif provider_name == "CoreMLExecutionProvider":
            params["coreml_flags"] = 2 if model_bits == 'FP16' else 0
        
        elif provider_name == "CANNExecutionProvider":
            params.update({
                "device_id": device_id,
                "precision_mode": f"force_{model_bits.lower()}"
            })
        
        return (provider_name, params) if params else provider_name
    
    def _init_openvino(self):
        try:
            from openvino import Core
            core = Core()
            device = ENGINEtoBackend[self._engine]
            if Path(self._weights_path).suffix == '.onnx':
                model = core.read_model(self._weights_path)
            elif Path(self._weights_path).suffix == '.xml' and Path(self._weights_path).with_suffix('.bin').exists():
                xml_path = Path(self._weights_path)
                bin_path = Path(self._weights_path).with_suffix('.bin')
                model = core.read_model(model=str(xml_path), weights=str(bin_path))
            else:
                raise ValueError("Invalid weights file format. Please provide either an ONNX or an XML+BIN file.")
            devices = ENGINEtoBackend["OpenVINO"]
            device = devices.get(self._device)
            self.model = core.compile_model(model=model, device_name=device)
            inputs = self.model.inputs
            for input_layer in inputs:
                input_shape = input_layer.shape
                if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                    self._input_width = input_shape[2]
                    self._input_height = input_shape[3]
            self._runtime_output_names = tuple(
                output.get_any_name() for output in self.model.outputs
            )
            # Compiled-model outputs always expose a partial shape, so unlike
            # OpenCV this probe cannot fail; dynamic dims become -1.
            self._runtime_output_shapes = {
                output.get_any_name(): _partial_shape_to_static(output.get_partial_shape())
                for output in self.model.outputs
            }
        except ImportError:
            raise ImportError("Please install openvino to use OpenVINO engine.")

    def _init_cann(self):
        try:
            from ais_bench.infer.interface import InferSession
            self.model = InferSession(0, self._weights_path)
        except ImportError:
            raise ImportError("Please install ais_bench to use CANN engine.")
        self._probe_cann_output_shapes()

    def _probe_cann_output_shapes(self) -> None:
        """Best-effort output probe for the Ascend backend.

        ``ais_bench``'s session API differs between releases, and a missing probe
        only costs the automatic schema-detection path -- the decoder validates
        shapes itself at decode time -- so every failure is swallowed.

        Names are recorded too, but only when nothing was configured: the
        ``output_names`` property prefers the artifact metadata, so a sidecar
        mapping still wins. Without this, an artifact that carries no sidecar
        would leave the decoder looking for its default tensor name in a mapping
        keyed by whatever the backend calls it.
        """
        try:
            outputs = list(self.model.get_outputs())
        except Exception:
            return
        shapes: Dict[str, Tuple[int, ...]] = {}
        names = []
        for index, output in enumerate(outputs):
            name = getattr(output, "name", None) or f"output_{index}"
            names.append(str(name))
            shape = getattr(output, "shape", None)
            if shape is not None:
                shapes[str(name)] = _static_shape(shape)
        if shapes:
            self._runtime_output_shapes = shapes
        if names and not self._configured_output_names:
            self._runtime_output_names = tuple(names)
    
    def _init_tensorrt(self):
        try:
            import tensorrt as trt
            from .utils import allocate_buffers
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            # Step 1: Load or build the TensorRT engine
            if Path(self._weights_path).suffix == '.onnx':
                self.model = self._build_engine_from_onnx(TRT_LOGGER)
            else:
                self.model = self._build_engine_from_engine(TRT_LOGGER)
            self.context = self.model.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.model)

            output_names = []
            output_shapes = {}
            for index in range(self.model.num_io_tensors):
                tensor_name = self.model.get_tensor_name(index)
                if self.model.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                    output_names.append(tensor_name)
                    output_shapes[tensor_name] = tuple(
                        int(dimension)
                        for dimension in self.model.get_tensor_shape(tensor_name)
                    )
            self._runtime_output_names = tuple(output_names)
            self._runtime_output_shapes = output_shapes
            
            input_layer_name = self.model.get_tensor_name(0)
            input_layer_shape = self.model.get_tensor_shape(input_layer_name)
            self._input_width = input_layer_shape[2]
            self._input_height = input_layer_shape[3]

        except ImportError:
            raise ImportError("Please install tensorrt and pycuda to use TensorRT engine.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {self._weights_path} not found.")
        except Exception as e:
            raise RuntimeError(f"TensorRT initialization failed: {str(e)}")

    def _build_engine_from_onnx(self, logger):
        """Build TensorRT engine from ONNX file (TensorRT 10.x API)."""
        try:
            import tensorrt as trt
            with trt.Builder(logger) as builder, builder.create_network(0) as network, \
                    builder.create_builder_config() as config, trt.OnnxParser(network, logger) as parser, \
                    trt.Runtime(logger) as runtime:                
                # Set the workspace memory limit
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
                with open(self._weights_path, "rb") as model:
                    print("Beginning ONNX file parsing")
                    if not parser.parse(model.read()):
                        print("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)
                return engine
        except ImportError:
            raise ImportError("Please install tensorrt and pycuda to use TensorRT engine.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {self._weights_path} not found.")
        except Exception as e:
            raise RuntimeError(f"TensorRT engine build failed: {str(e)}")

    def _build_engine_from_engine(self, logger):
        """Load pre-built TensorRT engine."""
        try:
            import tensorrt as trt
            with open(self._weights_path, "rb") as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                return engine
        except ImportError:
            raise ImportError("Please install tensorrt and pycuda to use TensorRT engine.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {self._weights_path} not found.")
        except Exception as e:
            raise RuntimeError(f"TensorRT engine load failed: {str(e)}")

    def _init_coreml(self):
        try: 
            import coremltools as ct
            self.model = ct.models.MLModel(self._weights_path)
            input_description = self.model.get_spec().description.input[0]
            self.input_name = input_description.name
            self._runtime_output_names = tuple(
                output.name for output in self.model.get_spec().description.output
            )
            self._runtime_output_shapes = {}
            for output in self.model.get_spec().description.output:
                multi_array = getattr(output.type, "multiArrayType", None)
                if multi_array is None or not multi_array.shape:
                    continue
                self._runtime_output_shapes[output.name] = tuple(
                    int(dimension) for dimension in multi_array.shape
                )
            input_type = input_description.type
            if hasattr(input_type, 'imageType'):
                shape = input_type.imageType
                self._input_width = shape.width
                self._input_height = shape.height
        except ImportError:
            raise ImportError("Please install coremltools to use CoreML engine.")

    
    def preprocess(self, input_image):
        """
        Preprocess a frame using the established live-inference input contract.

        Returns the model-ready image and the inverse affine transform.
        """
        from animalposetracker.preprocessing.live import preprocess_live_frame

        return preprocess_live_frame(
            input_image,
            self._input_width,
            self._input_height,
            coreml=self._coreml,
        )

    def inference(self, img):
        # Run inference using the preprocessed image data
        if self._engine == 'OpenCV':
            self.model.setInput(img)
            pred = self.model.forward()
        elif self._engine == 'OpenVINO':
            pred = self.model([img])
        elif self._engine == 'ONNX':
            pred = self.model.run([self.output_name], {self.input_name: img})
        elif self._engine == 'TensorRT':
            pred = self.inference_tensorrt(img)
        elif self._engine == 'CANN':
            pred = self.model.infer([img])
        elif self._engine == 'CoreML':
            pred = self.model.predict({self.input_name: img})
        
        return pred

    def inference_named_outputs(self, img, output_names=None):
        """Run the model and normalize its outputs to a name-to-array mapping.

        The legacy ``inference`` method remains unchanged for the existing GUI
        path. Stream pipelines use this method to consume multi-output models.
        """
        from animalposetracker.postprocessing.output_decoders import normalize_output_tensors

        if self.model is None:
            raise RuntimeError("Model not loaded, call model_init() first.")

        requested_names = tuple(output_names or self.output_names or ())
        if self._engine == 'OpenCV':
            self.model.setInput(img)
            layer_names = list(requested_names or self._runtime_output_names)
            if len(layer_names) > 1:
                raw_outputs = self.model.forward(layer_names)
            elif layer_names:
                raw_outputs = self.model.forward(layer_names[0])
            else:
                raw_outputs = self.model.forward()
        elif self._engine == 'OpenVINO':
            raw_outputs = self.model([img])
        elif self._engine == 'ONNX':
            names = list(requested_names or self._runtime_output_names)
            raw_outputs = self.model.run(
                names or None,
                {self.input_name: img},
            )
        elif self._engine == 'TensorRT':
            raw_outputs = self.inference_tensorrt(img)
            names = requested_names or self._runtime_output_names
            shaped_outputs = []
            for index, output in enumerate(raw_outputs):
                name = names[index] if index < len(names) else None
                shape = self._runtime_output_shapes.get(name)
                output = np.asarray(output)
                if shape and all(dimension >= 0 for dimension in shape):
                    if output.size == int(np.prod(shape)):
                        output = output.reshape(shape)
                shaped_outputs.append(output)
            raw_outputs = shaped_outputs
        elif self._engine == 'CANN':
            raw_outputs = self.model.infer([img])
        elif self._engine == 'CoreML':
            raw_outputs = self.model.predict({self.input_name: img})
        else:
            raise ValueError(f"Unsupported inference engine: {self._engine!r}")

        names = requested_names or self._runtime_output_names
        normalized = normalize_output_tensors(raw_outputs, names)
        if self._engine == 'OpenCV' and not self._runtime_output_shapes:
            # Pre-forward shape probing is best-effort on OpenCV (see
            # _probe_opencv_output_shapes). Backfill from what the graph actually
            # produced, so the mapping is available from the second frame on.
            self._runtime_output_shapes = {
                str(name): tuple(int(dimension) for dimension in np.asarray(array).shape)
                for name, array in normalized.items()
            }
        return normalized
    
    def inference_tensorrt(self, img):
        try:
            from .utils import do_inference

            # Copy the input image to the device
            self.inputs[0].host = img

            trt_outputs = do_inference(
                context=self.context,
                engine=self.model,
                bindings=self.bindings,
                inputs=self.inputs,
                outputs=self.outputs,
                stream=self.stream
            )
            return trt_outputs
        except ImportError:
            raise ImportError("Please install tensorrt and pycuda to use TensorRT engine.")

    def draw_detections(self, 
                        img, 
                        box, 
                        score, 
                        class_id, 
                        line_width=2, 
                        bbox=False, 
                        classes=False):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.classes_color_palette[class_id]

        if bbox:

            # Draw the bounding box on the image
            cv2.rectangle(
                img=img,
                pt1=(int(x1), int(y1)),
                pt2=(int(x1 + w), int(y1 + h)),
                color=color,
                thickness=line_width,
                lineType=cv2.LINE_AA
            )

        if classes:
            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {score:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = int(x1)
            label_y = int(y1 - 10) if y1 - 10 > label_height else int(y1 + 10)

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            font_scale = self._input_height /  960.0 * self.visualize_config['font_scale']
            thickness = max(1, int(font_scale))

            cv2.putText(img, label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness, 
                        cv2.LINE_AA)

    def draw_keypoints(self, img, keypoints, radius=5):
        for i, keypoint in enumerate(keypoints):
            x, y = keypoint[0], keypoint[1]
            color_k = [int(x) for x in self.keypoints_color_palette[i]]

            if x != 0 and y != 0:
                cv2.circle(img,
                           (int(x), int(y)),
                           radius=radius,
                           color=color_k,
                           thickness=-1,
                           lineType=cv2.LINE_AA)

    def draw_skeleton(self, img, keypoints, line_width=1):
        for i, sk in enumerate(self.skeleton):
            pos1 = (int(keypoints[(sk[0]), 0]), int(keypoints[(sk[0]), 1]))
            pos2 = (int(keypoints[(sk[1]), 0]), int(keypoints[(sk[1]), 1]))

            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                continue

            cv2.line(img,
                     pos1, pos2,
                     color=self.skeleton_color_palette[i],
                     thickness=line_width,
                     lineType=cv2.LINE_AA
                     )
    
    def visualize(self, frame, results):
        """
        Visualizes the output of the model on a single frame of video.

        Args:
            frame: The input frame to process.
            results: The output of the model on the input frame.

        Returns:
            output_img: The output image with drawn detections.
        """

        boxes = results['boxes']
        keypoints_list = results['keypoints_list']
        scores = results['scores']
        class_ids = results['class_ids']

        if self.visualize_config['background'] == 'Black':
            frame = np.zeros_like(frame)
        elif self.visualize_config['background'] == 'White':
            frame = np.ones_like(frame) * 255
        elif self.visualize_config['background'] == 'Original':
            pass

        if len(boxes) > 0:
            if boxes.ndim == 1:
                self.draw_detections(frame, boxes, scores, class_ids, 
                                     bbox=self.visualize_config['show_bbox'],
                                     classes=self.visualize_config['show_classes'])
                if self.visualize_config['show_keypoints']:
                    self.draw_keypoints(frame, keypoints_list,
                                        radius=self.visualize_config['radius'])
                if self.visualize_config['show_skeletons']:
                    self.draw_skeleton(frame, keypoints_list,
                                       line_width=self.visualize_config['skeleton_line_width'])
            else:
                for i, box in enumerate(boxes):
                    self.draw_detections(frame, box, scores[i], class_ids[i],
                                        bbox=self.visualize_config['show_bbox'],
                                        classes=self.visualize_config['show_classes'])
                    if self.visualize_config['show_keypoints']:
                        self.draw_keypoints(frame, keypoints_list[i],
                                            radius=self.visualize_config['radius'])
                    if self.visualize_config['show_skeletons']:
                        self.draw_skeleton(frame, keypoints_list[i],
                                           line_width=self.visualize_config['skeleton_line_width'])

        times = {
            'preprocess_time': 'preprocess: {:.3f} ms'.format(results['preprocess_time'] * 1000.0),
            'inference_time': 'inference: {:.3f} ms'.format(results['inference_time'] * 1000.0),
            'postprocess_time': 'postprocess: {:.3f} ms'.format(results['postprocess_time'] * 1000.0),
            'fps': 'FPS: {:.1f} FPS'.format(results['fps'])
        }

        font_scale = self._input_height /  960.0 * self.visualize_config['font_scale']
        thickness = max(1, int(font_scale))
        
        line_height = int(self._input_height * 0.04 * self.visualize_config['font_scale'])
        x, y = int(self._input_width * 0.02), int(self._input_height * 0.05)

        font_color=(0, 255, 0)

        if self.visualize_config['show_fps']:
            cv2.putText(frame, 
                        f"FPS: {times['fps']}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        font_color, 
                        thickness, 
                        cv2.LINE_AA)
            y += line_height

        if self.visualize_config['show_preprocess_time']:
            cv2.putText(frame, 
                        f"Preprocess: {times['preprocess_time']}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, font_color, thickness, cv2.LINE_AA)
            y += line_height

        if self.visualize_config['show_inference_time']:
            cv2.putText(frame, 
                        f"Inference: {times['inference_time']}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, font_color, thickness, cv2.LINE_AA)
            y += line_height
        
        if self.visualize_config['show_postprocess_time']:
            cv2.putText(frame, 
                        f"Postprocess: {times['postprocess_time']}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, font_color, thickness, cv2.LINE_AA)
        
        return frame
