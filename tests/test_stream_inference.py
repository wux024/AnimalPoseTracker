import unittest
from unittest.mock import patch

import numpy as np

from animalposetracker.postprocessing.output_decoders import (
    RTMO_MMDEPLOY,
    RTMPOSE_SIMCC,
    YOLO_POSE_END2END,
    YOLO_POSE_RAW,
    PoseOutputConfig,
    decode_pose_outputs,
    decode_simcc,
    normalize_output_tensors,
)
from animalposetracker.inference.stream_inference import (
    DetectorBox,
    StreamInferencePipeline,
    StreamInputConfig,
)
from animalposetracker.inference.inferencer import InferenceEngine


class OutputNormalizationTests(unittest.TestCase):
    def test_single_array_is_assigned_configured_name(self):
        result = normalize_output_tensors(np.ones((1, 2)), ("predictions",))
        self.assertEqual(list(result), ["predictions"])
        self.assertEqual(result["predictions"].shape, (1, 2))

    def test_multi_output_mapping_is_reordered_by_name(self):
        result = normalize_output_tensors(
            {"simcc_y": np.full((1, 1, 8), 2), "simcc_x": np.full((1, 1, 8), 1)},
            ("simcc_x", "simcc_y"),
        )
        self.assertEqual(list(result), ["simcc_x", "simcc_y"])
        self.assertEqual(float(result["simcc_x"].max()), 1.0)
        self.assertEqual(float(result["simcc_y"].max()), 2.0)

    def test_mapping_with_backend_port_keys_uses_declared_order(self):
        class OutputPort:
            def __init__(self, name):
                self.any_name = name

        key_a = OutputPort("port_a")
        key_b = OutputPort("port_b")
        outputs = {key_a: np.asarray([1]), key_b: np.asarray([2])}
        result = normalize_output_tensors(outputs, ("dets", "pred_kpts"))
        self.assertEqual(list(result), ["dets", "pred_kpts"])
        self.assertEqual(int(result["dets"][0]), 1)
        self.assertEqual(int(result["pred_kpts"][0]), 2)


class RuntimeOutputAdapterTests(unittest.TestCase):
    class FakeOpenCV:
        def setInput(self, _input):
            pass

        def forward(self, _names=None):
            return [np.asarray([[1.0]]), np.asarray([[2.0]])]

    class FakeONNX:
        def run(self, names, _feeds):
            self.requested_names = names
            return [np.asarray([[1.0]]), np.asarray([[2.0]])]

    class FakeOpenVINO:
        def __call__(self, _inputs):
            return {"dets": np.asarray([[1.0]]), "pred_kpts": np.asarray([[2.0]])}

    class FakeCANN:
        def infer(self, _inputs):
            return [np.asarray([[1.0]]), np.asarray([[2.0]])]

    class FakeCoreML:
        def predict(self, _inputs):
            return {"simcc_x": np.asarray([[1.0]]), "simcc_y": np.asarray([[2.0]])}

    def _make_engine(self, name, model, output_names):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = model
        engine._engine = name
        engine._input_name = "images"
        engine.input_name = "images"
        engine._configured_output_names = tuple(output_names)
        engine._runtime_output_names = tuple(output_names)
        engine._runtime_output_shapes = {}
        return engine

    def _assert_two_named_outputs(self, engine):
        result = engine.inference_named_outputs(np.zeros((1, 3, 4, 4), dtype=np.float32))
        self.assertEqual(list(result), ["dets", "pred_kpts"])
        self.assertEqual(float(result["dets"][0, 0]), 1.0)
        self.assertEqual(float(result["pred_kpts"][0, 0]), 2.0)

    def test_opencv_onnx_openvino_cann_and_coreml_multi_outputs(self):
        self._assert_two_named_outputs(self._make_engine(
            "OpenCV", self.FakeOpenCV(), ("dets", "pred_kpts")
        ))
        self._assert_two_named_outputs(self._make_engine(
            "ONNX", self.FakeONNX(), ("dets", "pred_kpts")
        ))
        self._assert_two_named_outputs(self._make_engine(
            "OpenVINO", self.FakeOpenVINO(), ("dets", "pred_kpts")
        ))
        self._assert_two_named_outputs(self._make_engine(
            "CANN", self.FakeCANN(), ("dets", "pred_kpts")
        ))
        coreml = self._make_engine(
            "CoreML", self.FakeCoreML(), ("simcc_x", "simcc_y")
        )
        result = coreml.inference_named_outputs(np.zeros((1, 3, 4, 4), dtype=np.float32))
        self.assertEqual(list(result), ["simcc_x", "simcc_y"])

    def test_tensorrt_flat_outputs_are_reshaped_to_engine_tensor_shape(self):
        engine = self._make_engine("TensorRT", object(), ("dets", "pred_kpts"))
        engine._runtime_output_shapes = {
            "dets": (1, 2, 1),
            "pred_kpts": (1, 2, 1),
        }
        with patch.object(
            engine,
            "inference_tensorrt",
            return_value=[np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])],
        ):
            result = engine.inference_named_outputs(
                np.zeros((1, 3, 4, 4), dtype=np.float32)
            )
        self.assertEqual(result["dets"].shape, (1, 2, 1))
        self.assertEqual(result["pred_kpts"].shape, (1, 2, 1))


class PoseOutputDecoderTests(unittest.TestCase):
    def test_yolo_raw_decodes_boxes_and_suppresses_duplicate_candidates(self):
        config = PoseOutputConfig(
            schema=YOLO_POSE_RAW,
            num_classes=1,
            num_keypoints=2,
            keypoint_dims=3,
        )
        # [cx, cy, w, h, class_score, 2 * (x, y, visibility)] x candidates
        candidates = np.asarray([
            [10, 10, 8, 8, 0.9, 4, 5, 0.8, 6, 7, 0.7],
            [10, 10, 8, 8, 0.7, 5, 6, 0.6, 7, 8, 0.5],
        ], dtype=np.float32)
        outputs = {"predictions": candidates.T[None]}

        result = decode_pose_outputs(outputs, config)

        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].bbox_xyxy, [6, 6, 14, 14])
        np.testing.assert_allclose(result[0].keypoints_xy, [[4, 5], [6, 7]])
        np.testing.assert_allclose(result[0].keypoints_visible, [0.8, 0.7])

    def test_yolo_end2end_does_not_apply_external_nms(self):
        config = PoseOutputConfig(
            schema=YOLO_POSE_END2END,
            num_classes=1,
            num_keypoints=1,
            keypoint_dims=3,
            confidence_threshold=0.0,
        )
        outputs = {"predictions": np.asarray([[[0, 0, 10, 10, 0.9, 0, 4, 5, 0.8],
                                                  [0, 0, 10, 10, 0.8, 0, 5, 6, 0.7],
                                                  [0, 0, 0, 0, 0.0, 0, 0, 0, 0]]], dtype=np.float32)}

        with patch("animalposetracker.postprocessing.output_decoders.cv2.dnn.NMSBoxes") as nms:
            result = decode_pose_outputs(outputs, config)

        self.assertEqual(len(result), 2)
        nms.assert_not_called()

    def test_rtmo_mmdeploy_decodes_dets_and_keypoints_as_a_pair(self):
        config = PoseOutputConfig(
            schema=RTMO_MMDEPLOY,
            num_classes=1,
            num_keypoints=2,
            keypoint_dims=3,
            output_names=("dets", "pred_kpts"),
        )
        outputs = {
            "dets": np.asarray([[[10, 20, 30, 40, 0.9]]], dtype=np.float32),
            "pred_kpts": np.asarray([[[[11, 21, 0.8], [25, 35, 0.6]]]], dtype=np.float32),
        }
        inverse = np.asarray([[1, 0, -2], [0, 1, -3]], dtype=np.float32)

        result = decode_pose_outputs(outputs, config, inverse_affine=inverse)

        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].bbox_xyxy, [8, 17, 28, 37])
        np.testing.assert_allclose(result[0].keypoints_xy, [[9, 18], [23, 32]])
        np.testing.assert_allclose(result[0].keypoints_visible, [0.8, 0.6])

    def test_simcc_decodes_split_axes_and_inverse_transform(self):
        config = PoseOutputConfig(
            schema=RTMPOSE_SIMCC,
            num_classes=1,
            num_keypoints=1,
            keypoint_dims=2,
            output_names=("simcc_x", "simcc_y"),
            simcc_split_ratio=2.0,
        )
        simcc_x = np.zeros((1, 1, 16), dtype=np.float32)
        simcc_y = np.zeros((1, 1, 16), dtype=np.float32)
        simcc_x[0, 0, 8] = 0.9
        simcc_y[0, 0, 6] = 0.8
        inverse = np.asarray([[1, 0, 10], [0, 1, 20]], dtype=np.float32)

        result = decode_simcc(
            {"simcc_x": simcc_x, "simcc_y": simcc_y},
            config,
            inverse_affine=inverse,
            bbox_xyxy=[5, 6, 15, 16],
            bbox_score=0.7,
            class_id=2,
        )

        np.testing.assert_allclose(result.keypoints_xy, [[14, 23]])
        np.testing.assert_allclose(result.keypoint_scores, [0.8])
        np.testing.assert_allclose(result.bbox_xyxy, [5, 6, 15, 16])
        self.assertEqual(result.class_id, 2)


class StreamPipelineTests(unittest.TestCase):
    class FakePoseEngine:
        def __init__(self, outputs, width=8, height=8):
            self.model = object()
            self.input_width = width
            self.input_height = height
            self._outputs = outputs
            self.calls = []

        @property
        def output_names(self):
            return ()

        def inference_named_outputs(self, model_input, output_names=None):
            self.calls.append(np.asarray(model_input).copy())
            return self._outputs

    class FakeDetector:
        def __init__(self, boxes):
            self.boxes = boxes

        def predict_boxes(self, frame):
            del frame
            return self.boxes

    def test_topdown_pipeline_requires_and_uses_detector_provider(self):
        config = PoseOutputConfig(
            schema=RTMPOSE_SIMCC,
            num_keypoints=1,
            keypoint_dims=2,
            simcc_split_ratio=2.0,
        )
        simcc_x = np.zeros((1, 1, 16), dtype=np.float32)
        simcc_y = np.zeros((1, 1, 16), dtype=np.float32)
        simcc_x[0, 0, 8] = 1.0
        simcc_y[0, 0, 8] = 1.0
        engine = self.FakePoseEngine({"simcc_x": simcc_x, "simcc_y": simcc_y})
        with self.assertRaisesRegex(ValueError, "detector provider"):
            StreamInferencePipeline(
                engine,
                config,
                input_mode="topdown",
                input_config=StreamInputConfig(input_size=(8, 8)),
            )

        pipeline = StreamInferencePipeline(
            engine,
            config,
            input_mode="topdown",
            input_config=StreamInputConfig(
                input_size=(8, 8),
                color_order="BGR",
                input_scale=1.0,
                pixel_mean=(0, 0, 0),
                pixel_std=(1, 1, 1),
                bbox_padding=1.0,
            ),
            detector=self.FakeDetector([
                DetectorBox([10, 10, 30, 30], 0.9, 0),
                DetectorBox([0, 0, 20, 20], 0.8, 1),
            ]),
        )

        result = pipeline.process_frame(np.zeros((40, 40, 3), dtype=np.uint8))

        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].bbox_xyxy, [10, 10, 30, 30])
        np.testing.assert_allclose(result[0].keypoints_xy, [[20, 20]], atol=1e-5)
        np.testing.assert_allclose(result[1].bbox_xyxy, [0, 0, 20, 20])
        np.testing.assert_allclose(result[1].keypoints_xy, [[10, 10]], atol=1e-5)
        self.assertEqual(len(engine.calls), 2)

    def test_topdown_pipeline_returns_empty_without_running_pose_model(self):
        config = PoseOutputConfig(
            schema=RTMPOSE_SIMCC,
            num_keypoints=1,
            keypoint_dims=2,
        )
        engine = self.FakePoseEngine({})
        pipeline = StreamInferencePipeline(
            engine,
            config,
            input_mode="topdown",
            input_config=StreamInputConfig(input_size=(8, 8)),
            detector=self.FakeDetector([]),
        )

        self.assertEqual(pipeline.process_frame(np.zeros((40, 40, 3), dtype=np.uint8)), [])
        self.assertEqual(engine.calls, [])

    def test_rtmo_pipeline_uses_mmpose_bottomup_preprocessing(self):
        config = PoseOutputConfig(
            schema=RTMO_MMDEPLOY,
            num_keypoints=1,
            keypoint_dims=3,
        )
        engine = self.FakePoseEngine({
            "dets": np.asarray([[[0, 0, 8, 8, 0.9]]], dtype=np.float32),
            "pred_kpts": np.asarray([[[[4, 4, 0.8]]]], dtype=np.float32),
        })
        pipeline = StreamInferencePipeline(
            engine,
            config,
            input_mode="full_frame",
            input_config=StreamInputConfig(
                input_size=(8, 8),
                preprocess_mode="mmpose_bottomup",
                color_order="BGR",
                input_scale=1.0,
                pixel_mean=(0, 0, 0),
                pixel_std=(1, 1, 1),
            ),
        )

        result = pipeline.process_frame(np.zeros((4, 8, 3), dtype=np.uint8))

        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].keypoints_xy, [[4, 2]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
