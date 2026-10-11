"""Tests for the inferencer's per-engine output-shape collection.

The schema resolver can only read a layout off the runtime shapes if the shapes
are actually there. They were previously collected on TensorRT alone, which left
ONNX -- the deployment path that matters most -- relying on the resolver's blind
fallback. These tests pin the collection on every engine that can report shapes,
including the OpenCV path, whose probe is best-effort and therefore has to have a
working fallback.
"""

import contextlib
import importlib
import inspect
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from animalposetracker.inference.inferencer import (
    InferenceEngine,
    _partial_shape_to_static,
    _static_shape,
)
from animalposetracker.postprocessing.output_decoders import YOLO_POSE_RAW


KPT, DIMS = 17, 3
PROCESSED = 6 + KPT * DIMS  # 57


def divert_stdout():
    """Swallow the session banners the engine prints during init."""
    return contextlib.redirect_stdout(io.StringIO())


def bare_engine(**attributes):
    """An ``InferenceEngine`` without ``__init__``, so no backend is required."""
    engine = object.__new__(InferenceEngine)
    engine._configured_output_names = ()
    engine._runtime_output_names = ()
    engine._runtime_output_shapes = {}
    engine.output_name = None
    for key, value in attributes.items():
        setattr(engine, key, value)
    return engine


class StaticShapeTests(unittest.TestCase):
    def test_plain_ints_pass_through(self):
        self.assertEqual(_static_shape((1, PROCESSED, 8400)), (1, PROCESSED, 8400))

    def test_negative_sentinels_become_minus_one(self):
        self.assertEqual(_static_shape((1, -1, 8400)), (1, -1, 8400))
        self.assertEqual(_static_shape((-7,)), (-1,))

    def test_symbolic_dimensions_become_minus_one(self):
        # ONNX Runtime hands back a str for a symbolically named dimension.
        self.assertEqual(_static_shape(("batch", PROCESSED, "anchors")), (-1, PROCESSED, -1))
        self.assertEqual(_static_shape((None, PROCESSED)), (-1, PROCESSED))

    def test_integer_like_scalars_are_static(self):
        self.assertEqual(_static_shape((np.int64(1), np.int32(PROCESSED))), (1, PROCESSED))

    def test_booleans_are_not_dimensions(self):
        self.assertEqual(_static_shape((True, False)), (-1, -1))

    def test_floats_are_not_dimensions(self):
        self.assertEqual(_static_shape((1.0,)), (-1,))

    def test_empty_shape(self):
        self.assertEqual(_static_shape(()), ())

    def test_numpy_shape_property(self):
        self.assertEqual(_static_shape(np.zeros((1, PROCESSED, 8400)).shape), (1, PROCESSED, 8400))


class _FakeDimension:
    def __init__(self, length=None, static=True):
        self._length = length
        self.is_static = static

    def get_length(self):
        if self._length is None:
            raise RuntimeError("dynamic dimension has no length")
        return self._length


class _FakePartialShape:
    def __init__(self, dimensions, rank_static=True):
        self._dimensions = list(dimensions)
        self.rank = types.SimpleNamespace(is_static=rank_static)

    def __iter__(self):
        return iter(self._dimensions)


class PartialShapeTests(unittest.TestCase):
    def test_static_shape(self):
        partial = _FakePartialShape([_FakeDimension(1), _FakeDimension(PROCESSED)])
        self.assertEqual(_partial_shape_to_static(partial), (1, PROCESSED))

    def test_dynamic_dimension_becomes_minus_one(self):
        partial = _FakePartialShape([_FakeDimension(1), _FakeDimension(static=False)])
        self.assertEqual(_partial_shape_to_static(partial), (1, -1))

    def test_dynamic_rank_is_empty(self):
        partial = _FakePartialShape([_FakeDimension(1)], rank_static=False)
        self.assertEqual(_partial_shape_to_static(partial), ())

    def test_unexpected_object_is_empty(self):
        self.assertEqual(_partial_shape_to_static(object()), ())

    def test_get_length_failure_is_empty(self):
        partial = _FakePartialShape([_FakeDimension(None)])
        self.assertEqual(_partial_shape_to_static(partial), ())


class _FakeCvDnnModel:
    """Mimics the two OpenCV DNN calls the probe makes."""

    def __init__(self, layer_ids, layer_shapes, fail_probe=False):
        self._layer_ids = layer_ids
        self._layer_shapes = layer_shapes
        self._fail_probe = fail_probe

    def getLayersShapes(self, input_shape):
        if self._fail_probe:
            raise RuntimeError("shape inference unavailable on this build")
        return (input_shape, [], self._layer_shapes)

    def getLayerId(self, name):
        return self._layer_ids.get(name, -1)


class OpenCVProbeTests(unittest.TestCase):
    def _engine(self, model, names=("predictions",)):
        return bare_engine(
            model=model,
            _engine="OpenCV",
            _input_width=640,
            _input_height=640,
            _runtime_output_names=names,
        )

    def test_shapes_are_probed(self):
        layer_shapes = [None] * 5 + [[(1, PROCESSED, 8400)]]
        engine = self._engine(_FakeCvDnnModel({"predictions": 5}, layer_shapes))
        self.assertEqual(engine._probe_opencv_output_shapes(), {"predictions": (1, PROCESSED, 8400)})

    def test_multiple_outputs_are_probed_independently(self):
        layer_shapes = [None, [(1, PROCESSED, 8400)], None, [(1, 8400, PROCESSED)]]
        model = _FakeCvDnnModel({"a": 1, "b": 3}, layer_shapes)
        engine = self._engine(model, names=("a", "b"))
        self.assertEqual(
            engine._probe_opencv_output_shapes(),
            {"a": (1, PROCESSED, 8400), "b": (1, 8400, PROCESSED)},
        )

    def test_unknown_layer_is_skipped(self):
        engine = self._engine(_FakeCvDnnModel({}, [[(1, PROCESSED, 8400)]]))
        self.assertEqual(engine._probe_opencv_output_shapes(), {})

    def test_layer_without_shapes_is_skipped(self):
        engine = self._engine(_FakeCvDnnModel({"predictions": 0}, [[]]))
        self.assertEqual(engine._probe_opencv_output_shapes(), {})

    def test_probe_failure_yields_empty_mapping(self):
        engine = self._engine(_FakeCvDnnModel({}, [], fail_probe=True))
        self.assertEqual(engine._probe_opencv_output_shapes(), {})


class _FakeCvForwardModel:
    def __init__(self, outputs):
        self._outputs = outputs
        self.last_layer = None

    def setInput(self, img):
        self.last_input = img

    def forward(self, layer=None):
        self.last_layer = layer
        return self._outputs


class OpenCVLazyBackfillTests(unittest.TestCase):
    """OpenCV cannot always probe before the first frame; the forward pass must fill in."""

    def _engine(self, outputs, names=("predictions",), shapes=None):
        return bare_engine(
            model=_FakeCvForwardModel(outputs),
            _engine="OpenCV",
            _runtime_output_names=names,
            _runtime_output_shapes={} if shapes is None else shapes,
        )

    def test_shapes_are_backfilled_from_the_first_forward_pass(self):
        tensor = np.zeros((1, PROCESSED, 8400), dtype=np.float32)
        engine = self._engine(tensor)
        engine.inference_named_outputs(np.zeros((1, 3, 640, 640), dtype=np.float32))
        self.assertEqual(engine._runtime_output_shapes, {"predictions": (1, PROCESSED, 8400)})

    def test_a_successful_probe_is_not_overwritten(self):
        # The probe's answer is authoritative: it exists before the first frame,
        # which is exactly when the schema has to be resolved.
        probed = (1, PROCESSED, 8400)
        engine = self._engine(
            np.zeros((1, PROCESSED, 300), dtype=np.float32),
            shapes={"predictions": probed},
        )
        engine.inference_named_outputs(np.zeros((1, 3, 640, 640), dtype=np.float32))
        self.assertEqual(engine._runtime_output_shapes, {"predictions": probed})

    def test_missing_model_is_reported(self):
        engine = bare_engine(model=None, _engine="OpenCV")
        with self.assertRaises(RuntimeError):
            engine.inference_named_outputs(np.zeros((1, 3, 640, 640), dtype=np.float32))


class _FakeAisBenchOutput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeAisBenchSession:
    def __init__(self, outputs, fail=False):
        self._outputs = outputs
        self._fail = fail

    def get_outputs(self):
        if self._fail:
            raise RuntimeError("session not started")
        return self._outputs


class CannProbeTests(unittest.TestCase):
    def test_shapes_and_names_are_recorded(self):
        model = _FakeAisBenchSession([
            _FakeAisBenchOutput("predictions", (1, PROCESSED, 8400)),
        ])
        engine = bare_engine(model=model)
        engine._probe_cann_output_shapes()
        self.assertEqual(engine._runtime_output_shapes, {"predictions": (1, PROCESSED, 8400)})
        self.assertEqual(engine._runtime_output_names, ("predictions",))

    def test_configured_names_win(self):
        model = _FakeAisBenchSession([_FakeAisBenchOutput("backend_tensor_0", (1, PROCESSED, 8400))])
        engine = bare_engine(model=model)
        engine._configured_output_names = ("predictions",)
        engine._probe_cann_output_shapes()
        self.assertEqual(engine._runtime_output_names, ())
        self.assertEqual(engine._runtime_output_shapes, {"backend_tensor_0": (1, PROCESSED, 8400)})

    def test_unnamed_output_gets_a_positional_fallback(self):
        model = _FakeAisBenchSession([types.SimpleNamespace(shape=(1, PROCESSED, 8400))])
        engine = bare_engine(model=model)
        engine._probe_cann_output_shapes()
        self.assertEqual(engine._runtime_output_names, ("output_0",))
        self.assertEqual(engine._runtime_output_shapes, {"output_0": (1, PROCESSED, 8400)})

    def test_probe_failure_leaves_state_untouched(self):
        engine = bare_engine(model=_FakeAisBenchSession([], fail=True))
        engine._probe_cann_output_shapes()
        self.assertEqual(engine._runtime_output_shapes, {})
        self.assertEqual(engine._runtime_output_names, ())


class _FakeOrtValueInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeOrtSession:
    def __init__(self, outputs):
        self._outputs = outputs

    def get_inputs(self):
        return [_FakeOrtValueInfo("images", (1, 3, 640, 640))]

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


class OnnxShapeCollectionTests(unittest.TestCase):
    def _fake_onnxruntime(self, session):
        module = types.ModuleType("onnxruntime")
        module.get_available_providers = lambda: [
            "DnnlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        module.SessionOptions = lambda: types.SimpleNamespace()
        module.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)
        module.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL=1)
        module.InferenceSession = lambda *args, **kwargs: session
        return module

    def test_static_and_symbolic_outputs_are_collected(self):
        session = _FakeOrtSession([
            _FakeOrtValueInfo("output0", (1, PROCESSED, 8400)),
            _FakeOrtValueInfo("output1", ("batch", PROCESSED, "anchors")),
        ])
        engine = bare_engine(_device="Intel CPU", _model_bits="FP32", _weights_path="model.onnx")
        with patch.dict(sys.modules, {"onnxruntime": self._fake_onnxruntime(session)}):
            with divert_stdout():
                engine._init_onnx()
        self.assertEqual(
            engine._runtime_output_shapes,
            {"output0": (1, PROCESSED, 8400), "output1": (-1, PROCESSED, -1)},
        )
        self.assertEqual(engine._runtime_output_names, ("output0", "output1"))
        self.assertEqual(engine.output_name, "output0")

    def test_input_size_is_taken_from_the_graph(self):
        session = _FakeOrtSession([_FakeOrtValueInfo("output0", (1, PROCESSED, 8400))])
        engine = bare_engine(
            _device="Intel CPU",
            _model_bits="FP32",
            _weights_path="model.onnx",
            _input_width=0,
            _input_height=0,
        )
        with patch.dict(sys.modules, {"onnxruntime": self._fake_onnxruntime(session)}):
            with divert_stdout():
                engine._init_onnx()
        self.assertEqual((engine._input_width, engine._input_height), (640, 640))


class _FakeOvOutput:
    def __init__(self, name, dimensions, rank_static=True):
        self._name = name
        self._partial = _FakePartialShape(dimensions, rank_static=rank_static)

    def get_any_name(self):
        return self._name

    def get_partial_shape(self):
        return self._partial


class _FakeOvCompiledModel:
    def __init__(self, outputs):
        self.inputs = [
            types.SimpleNamespace(shape=(1, 3, 640, 640)),
        ]
        self.outputs = outputs


class OpenVinoShapeCollectionTests(unittest.TestCase):
    def _fake_openvino(self, compiled):
        module = types.ModuleType("openvino")

        class Core:
            def read_model(self, path):
                return object()

            def compile_model(self, model=None, device_name=None):
                return compiled

        module.Core = Core
        return module

    def test_static_outputs_are_collected(self):
        compiled = _FakeOvCompiledModel([
            _FakeOvOutput("predictions", [_FakeDimension(1), _FakeDimension(PROCESSED), _FakeDimension(8400)]),
        ])
        engine = bare_engine(_engine="OpenVINO", _device="Intel CPU", _weights_path="model.onnx")
        with patch.dict(sys.modules, {"openvino": self._fake_openvino(compiled)}):
            engine._init_openvino()
        self.assertEqual(engine._runtime_output_shapes, {"predictions": (1, PROCESSED, 8400)})
        self.assertEqual(engine._runtime_output_names, ("predictions",))
        self.assertEqual((engine._input_width, engine._input_height), (640, 640))

    def test_dynamic_output_becomes_minus_one(self):
        compiled = _FakeOvCompiledModel([
            _FakeOvOutput("predictions", [_FakeDimension(1), _FakeDimension(static=False), _FakeDimension(8400)]),
        ])
        engine = bare_engine(_engine="OpenVINO", _device="Intel CPU", _weights_path="model.onnx")
        with patch.dict(sys.modules, {"openvino": self._fake_openvino(compiled)}):
            engine._init_openvino()
        self.assertEqual(engine._runtime_output_shapes, {"predictions": (1, -1, 8400)})


class _FakeMlMultiArray:
    def __init__(self, shape):
        self.shape = shape


class _FakeMlOutput:
    def __init__(self, name, shape):
        self.name = name
        self.type = types.SimpleNamespace(multiArrayType=_FakeMlMultiArray(shape))


class _FakeMlInput:
    def __init__(self, name):
        self.name = name
        # No imageType: the engine must then keep its configured input size.
        self.type = types.SimpleNamespace()


class _FakeMlModel:
    def __init__(self, outputs):
        self._spec = types.SimpleNamespace(
            description=types.SimpleNamespace(
                input=[_FakeMlInput("image")],
                output=outputs,
            )
        )

    def get_spec(self):
        return self._spec


class CoreMlShapeCollectionTests(unittest.TestCase):
    def _fake_coremltools(self, model):
        module = types.ModuleType("coremltools")
        module.models = types.SimpleNamespace(MLModel=lambda path: model)
        return module

    def test_shapes_are_collected(self):
        model = _FakeMlModel([_FakeMlOutput("predictions", [1, PROCESSED, 8400])])
        engine = bare_engine(_engine="CoreML", _weights_path="model.mlpackage")
        with patch.dict(sys.modules, {"coremltools": self._fake_coremltools(model)}):
            engine._init_coreml()
        self.assertEqual(engine._runtime_output_shapes, {"predictions": (1, PROCESSED, 8400)})
        self.assertEqual(engine._runtime_output_names, ("predictions",))

    def test_output_without_a_multi_array_shape_is_skipped(self):
        output = types.SimpleNamespace(name="predictions", type=types.SimpleNamespace())
        engine = bare_engine(_engine="CoreML", _weights_path="model.mlpackage")
        with patch.dict(sys.modules, {"coremltools": self._fake_coremltools(_FakeMlModel([output]))}):
            engine._init_coreml()
        self.assertEqual(engine._runtime_output_shapes, {})
        self.assertEqual(engine._runtime_output_names, ("predictions",))


class EngineCoverageTests(unittest.TestCase):
    """Every wired engine must populate the mapping the schema resolver reads."""

    #: Engines whose init can report shapes, and the probe they delegate to when
    #: the backend's own session API is not enough.
    _ENGINE_METHODS = (
        "_init_opencv",
        "_init_onnx",
        "_init_openvino",
        "_init_cann",
        "_init_coreml",
        "_init_tensorrt",
    )

    def _reachable_source(self, name):
        source = inspect.getsource(getattr(InferenceEngine, name))
        for probe in ("_probe_opencv_output_shapes", "_probe_cann_output_shapes"):
            if probe in source:
                source += inspect.getsource(getattr(InferenceEngine, probe))
        return source

    def test_each_engine_records_runtime_output_shapes(self):
        for name in self._ENGINE_METHODS:
            with self.subTest(method=name):
                self.assertIn("_runtime_output_shapes", self._reachable_source(name))

    def test_probe_helpers_warn_nothing_on_failure(self):
        # Both probes swallow backend errors by design: an empty mapping only
        # costs the automatic schema detection, which the resolver reports once.
        for probe in ("_probe_opencv_output_shapes", "_probe_cann_output_shapes"):
            with self.subTest(probe=probe):
                source = inspect.getsource(getattr(InferenceEngine, probe))
                self.assertIn("except Exception", source)


class SharedHelperTests(unittest.TestCase):
    """The two helpers the probes rely on have to stay backend-agnostic."""

    def test_static_shape_never_raises_on_hostile_input(self):
        self.assertEqual(_static_shape([object(), "x", None, -1, 0]), (-1, -1, -1, -1, 0))

    def test_partial_shape_never_raises_on_hostile_input(self):
        self.assertEqual(_partial_shape_to_static(None), ())


def _optional_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


_ONNX = _optional_import("onnx")
_ORT = _optional_import("onnxruntime")

# The probe is an Identity node, so it needs no modern IR feature. `onnx` and
# `onnxruntime` are versioned independently and a model written at the newest IR version
# cannot be loaded by an older runtime (onnx 1.23 writes IR 14; onnxruntime 1.21 reads up
# to IR 10), so pin the file to an old, broadly supported pair instead of inheriting the
# installed `onnx` defaults and erroring out on a version mismatch that has nothing to do
# with what this test asserts.
_PROBE_IR_VERSION = 8
_PROBE_OPSET = 13


@unittest.skipUnless(_ONNX is not None and _ORT is not None,
                     "requires the onnx and onnxruntime packages")
class OnnxEndToEndTests(unittest.TestCase):
    """A real ONNX file through a real session, ending at a resolved schema.

    The stubs above prove each branch in isolation; this proves the shapes ONNX
    Runtime actually reports are the ones the schema resolver can consume.
    """

    KPT, DIMS = 17, 3
    CHANNELS = 4 + KPT * DIMS + 2  # RAW with two classes

    def _write_model(self, directory):
        helper = _ONNX.helper
        tensor_type = _ONNX.TensorProto.FLOAT
        # The engine reads its input as NCHW, so the probe keeps a 4-D input and reshapes it
        # to the YOLO-Pose RAW layout the resolver decodes. Reshape requires the element
        # counts to agree, which is what fixes the input area: CHANNELS * 8400 == 3 * 400 * 399.
        input_shape = [1, 3, 400, 399]
        shape_initializer = helper.make_tensor(
            "target_shape", _ONNX.TensorProto.INT64, [3], [1, self.CHANNELS, 8400]
        )
        node = helper.make_node("Reshape", ["images", "target_shape"], ["output0"])
        graph = helper.make_graph(
            [node],
            "shape_probe",
            [helper.make_tensor_value_info("images", tensor_type, input_shape)],
            [helper.make_tensor_value_info("output0", tensor_type, [1, self.CHANNELS, 8400])],
            [shape_initializer],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", _PROBE_OPSET)]
        )
        model.ir_version = _PROBE_IR_VERSION
        path = Path(directory) / "shape_probe.onnx"
        _ONNX.save(model, str(path))
        return path

    def test_reported_shapes_resolve_to_a_schema(self):
        from animalposetracker.inference.schema_resolution import resolve_output_schema

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_model(directory)
            engine = bare_engine(_device="Intel CPU", _model_bits="FP32", _weights_path=str(path))
            with divert_stdout():
                try:
                    engine._init_onnx()
                except RuntimeError as exc:
                    # A build of onnxruntime that refuses even the pinned IR version is an
                    # environment mismatch, not a regression in the resolver.
                    self.skipTest(f"installed onnxruntime cannot load the probe model: {exc}")

        self.assertEqual(engine._runtime_output_shapes["output0"], (1, self.CHANNELS, 8400))
        schema, num_classes = resolve_output_schema(
            output_shapes=engine._runtime_output_shapes,
            num_keypoints=self.KPT,
            keypoint_dims=self.DIMS,
        )
        self.assertEqual((schema, num_classes), (YOLO_POSE_RAW, 2))


if __name__ == "__main__":
    unittest.main()
