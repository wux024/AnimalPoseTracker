import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from animalposetracker.cfg import DATA_YAML_PATHS, MODEL_YAML_PATHS
from animalposetracker.nn import MODEL_YAML_PATHS as NN_MODEL_YAML_PATHS
from animalposetracker.prediction.cli import _output_stem
from animalposetracker.postprocessing.workflows import (
    filter_tracked_video_records,
    track_frame_predictions,
    triangulate_multiview_predictions,
)
from animalposetracker.pose3d import CameraCalibration
from animalposetracker.project.api import AnimalPoseTrackerProject
from animalposetracker.training.config import TrainingConfig
from animalposetracker.training.checkpoint import load_model_weights, save_checkpoint
from animalposetracker.data.pose import PoseTextDataset
from animalposetracker.training.engine import Trainer
from animalposetracker.training.losses import PoseDetectionLoss
from animalposetracker.data.simcc import SimCCLabel
from animalposetracker.evaluation.topdown import SimCCPoseValidator
from animalposetracker.nn.head import YOLOPoseHead
from animalposetracker.tracking import create_tracker
from animalposetracker.project.model_context import unique_output_directory


class ReviewFixTests(unittest.TestCase):
    def test_public_model_and_dataset_maps_resolve_to_files(self):
        self.assertEqual(MODEL_YAML_PATHS, NN_MODEL_YAML_PATHS)
        self.assertTrue(all(path.is_file() for path in MODEL_YAML_PATHS.values()))
        self.assertTrue(all(path.is_file() for path in DATA_YAML_PATHS.values()))

    def test_output_stems_are_stable_and_disambiguate_nested_sources(self):
        first = Path("C:/dataset/camera_a/frame001.png")
        second = Path("C:/dataset/camera_b/frame001.jpg")

        first_stem = _output_stem(first, [first, second])
        second_stem = _output_stem(second, [first, second])

        self.assertNotEqual(first_stem, second_stem)
        self.assertEqual(first_stem, _output_stem(first, [first, second]))

    def test_box_only_2d_yolo_labels_fail_instead_of_supervising_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (32, 32), color="white").save(image_dir / "sample.png")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            data_path = root / "dataset.yaml"
            data_path.write_text(yaml.safe_dump({
                "path": str(root),
                "train": "images/train",
                "kpt_shape": [1, 2],
                "names": {0: "mouse"},
            }), encoding="utf-8")

            dataset = PoseTextDataset(data_path, split="train", image_size=32)
            with self.assertRaisesRegex(ValueError, "no visibility mask"):
                dataset[0]

    def test_multiscale_targets_use_actual_input_width_and_height(self):
        head = SimpleNamespace(
            nc=1,
            kpt_shape=(1, 3),
            reg_max=16,
            stride=torch.tensor([8.0, 16.0, 32.0]),
        )
        criterion = PoseDetectionLoss(head, image_size=640, kpt_oks_sigmas=[0.1])
        self.assertEqual(criterion.assigner.stride_val, 16.0)
        tiny_box = torch.tensor([[[10.0, 10.0, 12.0, 12.0]]])
        nearby_anchor = torch.tensor([[4.0, 11.0]])
        self.assertTrue(criterion.assigner._centers_in_boxes(nearby_anchor, tiny_box)[0, 0, 0])
        targets = {
            "batch_indices": torch.tensor([0]),
            "classes": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
            "keypoints": torch.tensor([[[0.25, 0.75, 1.0]]]),
        }

        _labels, boxes, keypoints, _mask = criterion._prepare_targets(
            targets, batch_size=1, image_height=320, image_width=640
        )

        torch.testing.assert_close(boxes[0, 0], torch.tensor([256.0, 96.0, 384.0, 224.0]))
        torch.testing.assert_close(keypoints[0, 0, 0], torch.tensor([160.0, 240.0, 1.0]))

    def test_pose_head_rebuilds_cached_anchors_after_device_move(self):
        head = YOLOPoseHead(nc=1, kpt_shape=(1, 3), ch=(16, 32, 64)).eval()
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        feature_shapes = ((1, 16, 8, 8), (1, 32, 4, 4), (1, 64, 2, 2))

        with torch.inference_mode():
            head([torch.randn(shape) for shape in feature_shapes])
        self.assertEqual(head.anchors.device.type, "cpu")

        head.to("meta")
        meta_features = [torch.empty(shape, device="meta") for shape in feature_shapes]
        with torch.inference_mode():
            output, _raw = head(meta_features)

        self.assertEqual(head.anchors.device.type, "meta")
        self.assertEqual(head.strides.device.type, "meta")
        self.assertEqual(output.device.type, "meta")

    def test_simcc_label_masks_points_just_outside_the_crop(self):
        codec = SimCCLabel((64, 64), sigma=3.0, split_ratio=2.0)
        labels_x, _labels_y, weights = codec.encode(
            [[-1.0, 10.0]],
            [1.0],
        )

        self.assertEqual(weights.tolist(), [0.0])
        self.assertEqual(float(labels_x[0].sum()), 0.0)

    def test_simcc_oks_nms_groups_instances_by_image_like_mmpose(self):
        validator = object.__new__(SimCCPoseValidator)
        validator.kpt_oks_sigmas = np.asarray([0.1], dtype=np.float32)
        first = {
            "image_id": 1,
            "category_id": 1,
            "keypoints": [10.0, 10.0, 0.9],
            "score": 0.9,
            "area": 100.0,
        }
        second = {
            "image_id": 1,
            "category_id": 2,
            "keypoints": [10.0, 10.0, 0.9],
            "score": 0.8,
            "area": 100.0,
        }

        self.assertEqual(validator._oks_nms([first, second], threshold=0.5), [first])

    def test_prediction_records_can_be_tracked_and_filtered_by_track(self):
        tracker = create_tracker("bytetrack")
        raw = [{
            "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
            "score": 0.9,
            "class_id": 0,
            "class_name": "mouse",
            "keypoints": [[5.0, 5.0, 0.9], [10.0, 10.0, 0.9], [15.0, 15.0, 0.9]],
        }]
        first = track_frame_predictions(raw, tracker, 0, class_names=["mouse"])
        second = track_frame_predictions(raw, tracker, 1, class_names=["mouse"])
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

        records = [{
            "frame": frame,
            "predictions": [{
                "track_id": 1,
                "keypoints": [[x, 0.0, 0.9]],
            }],
        } for frame, x in enumerate([0.0, 0.0, 100.0, 0.0, 0.0])]
        filtered = filter_tracked_video_records(
            records, method="median", window_length=3, confidence_threshold=0.25
        )
        self.assertLess(filtered[2]["predictions"][0]["keypoints"][0][0], 10.0)
        self.assertEqual(records[2]["predictions"][0]["keypoints"][0][0], 100.0)

    def test_video_writer_keeps_raw_and_tracked_frame_records(self):
        from animalposetracker.prediction.cli import _write_video

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detection = {
            "bbox_xyxy": [0.0, 0.0, 20.0, 20.0],
            "score": 0.9,
            "class_id": 0,
            "keypoints": [[5.0, 5.0, 0.9], [10.0, 10.0, 0.9], [15.0, 15.0, 0.9]],
        }

        class FakeCapture:
            def __init__(self):
                self.frames = [frame.copy(), frame.copy()]

            def isOpened(self):
                return True

            def get(self, prop):
                return 30.0 if prop == 5 else 32

            def read(self):
                return (True, self.frames.pop(0)) if self.frames else (False, None)

            def release(self):
                pass

        class FakeWriter:
            def isOpened(self):
                return True

            def write(self, _frame):
                pass

            def release(self):
                pass

        raw_records = []
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "tracked.mp4"
            with patch("animalposetracker.prediction.cli.cv2.VideoCapture", return_value=FakeCapture()), \
                 patch("animalposetracker.prediction.cli.cv2.VideoWriter", return_value=FakeWriter()), \
                 patch("animalposetracker.prediction.cli.cv2.VideoWriter_fourcc", return_value=0):
                frame_count, tracked_records = _write_video(
                    Path(temporary) / "input.mp4",
                    output_path,
                    lambda image: (image.copy(), [dict(detection)]),
                    raw_records,
                    tracker=create_tracker("bytetrack"),
                    class_names=["mouse"],
                )

        self.assertEqual(frame_count, 2)
        self.assertNotIn("track_id", raw_records[0]["predictions"][0])
        self.assertEqual(
            [record["predictions"][0]["track_id"] for record in tracked_records],
            [1, 1],
        )

    def test_prediction_records_triangulate_a_single_animal_across_cameras(self):
        intrinsic = np.asarray([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        cameras = {
            "left": CameraCalibration("left", intrinsic, np.eye(3), [0.0, 0.0, 0.0]),
            "right": CameraCalibration("right", intrinsic, np.eye(3), [-1.0, 0.0, 0.0]),
        }
        world_point = np.asarray([[0.2, 0.15, 5.0]])
        records = {}
        for camera_id, local_track_id in (("left", 4), ("right", 17)):
            pixel = cameras[camera_id].project_points(world_point)[0]
            records[camera_id] = [{
                "source": f"{camera_id}.mp4",
                "frame": 0,
                "predictions": [{
                    "track_id": local_track_id,
                    "keypoints": [[pixel[0], pixel[1], 0.9]],
                }],
            }, {
                "source": "unrelated.mp4",
                "frame": 0,
                "predictions": [{
                    "track_id": local_track_id + 100,
                    "keypoints": [[pixel[0] + 100.0, pixel[1], 0.9]],
                }],
            }]

        with tempfile.TemporaryDirectory() as temporary:
            left_path = Path(temporary) / "left.json"
            right_path = Path(temporary) / "right.json"
            left_path.write_text(json.dumps(records["left"]), encoding="utf-8")
            right_path.write_text(json.dumps(records["right"]), encoding="utf-8")
            results = triangulate_multiview_predictions(
                {"left": left_path, "right": right_path},
                cameras,
                source_by_camera={"left": "left.mp4", "right": "right.mp4"},
            )

        np.testing.assert_allclose(
            results["animal_0"]["sequence"].points_3d[0, 0],
            world_point[0],
            atol=1e-5,
        )

    def test_save_false_is_preserved_in_typed_training_config(self):
        config = TrainingConfig.from_mapping({"save": False, "verbose": False})
        self.assertFalse(config.save)
        self.assertFalse(config.verbose)

    def test_save_false_skips_checkpoint_writes(self):
        trainer = object.__new__(Trainer)
        trainer.rank = 0
        trainer.config = SimpleNamespace(save=False)

        with patch("animalposetracker.training.engine.save_checkpoint") as save_checkpoint:
            trainer._save_checkpoint(None, None, None, None, 0, 0, None, 0, None)

        save_checkpoint.assert_not_called()

    def test_native_checkpoint_with_numpy_rng_state_loads_safely_as_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "native.pt"
            source = torch.nn.Linear(2, 2)
            target = torch.nn.Linear(2, 2)
            save_checkpoint(checkpoint, source)

            report = load_model_weights(checkpoint, target, strict=False)

        self.assertEqual(report["loaded_tensors"], 2)

    def test_unique_run_directory_does_not_replace_an_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "train"
            first.mkdir()

            second = unique_output_directory(first, exist_ok=False)

            self.assertEqual(second, Path(temporary) / "train2")
            self.assertTrue(first.is_dir())

    def test_coco_2d_dataset_rejects_unlabeled_landmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images" / "train"
            annotation_dir = root / "annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir()
            Image.new("RGB", (32, 32), color="white").save(image_dir / "sample.png")
            payload = {
                "images": [{"id": 1, "file_name": "images/train/sample.png", "width": 32, "height": 32}],
                "categories": [{"id": 1, "name": "mouse", "keypoints": ["nose"]}],
                "annotations": [{
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [4, 4, 10, 10],
                    "area": 100,
                    "keypoints": [0, 0, 0],
                    "num_keypoints": 0,
                }],
            }
            (annotation_dir / "train.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            data_path = root / "dataset.yaml"
            data_path.write_text(yaml.safe_dump({
                "annotation_format": "coco",
                "path": str(root),
                "train": "images/train",
                "train_annotations": "annotations/train.json",
                "kpt_shape": [1, 2],
                "names": {0: "mouse"},
            }), encoding="utf-8")

            dataset = PoseTextDataset(data_path, split="train", image_size=32)
            with self.assertRaisesRegex(ValueError, "unlabeled keypoints"):
                dataset[0]

    def test_project_source_copy_preserves_subfolders_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            first = source / "left" / "same.png"
            second = source / "right" / "same.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"left")
            second.write_bytes(b"right")
            project = AnimalPoseTrackerProject(
                local_path=root,
                project_name="source-test",
                worker="test",
                model_type="AnimalRTPose",
                model_scale="N",
            )
            project.create_project_dirs()

            project.add_source_to_project(source)
            project.add_source_to_project(source)

            self.assertEqual(len(project.project_config["sources"]), 1)
            copied_root = Path(project.project_config["sources"][0])
            self.assertEqual((copied_root / "left" / "same.png").read_bytes(), b"left")
            self.assertEqual((copied_root / "right" / "same.png").read_bytes(), b"right")

    def test_project_predict_exposes_video_tracking_and_filter_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = AnimalPoseTrackerProject(
                local_path=temporary,
                project_name="analysis-test",
                worker="test",
                model_type="AnimalRTPose",
                model_scale="N",
            )
            project.create_new_project()
            weights = project.project_path / "runs" / "train" / "weights" / "best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"placeholder")
            project.other_config["model"] = str(weights)

            with patch.object(project, "_execute_command") as execute:
                output_dir = project.predict(
                    inference_source="input.mp4",
                    tracker="bytetrack",
                    pose_filter="median",
                )

            command = execute.call_args.args[0]
            self.assertEqual(output_dir, project.prediction_output_dir)
            self.assertIn(str(output_dir), command)
            self.assertIn("--tracker", command)
            self.assertIn("bytetrack", command)
            self.assertIn("--pose-filter", command)
            self.assertIn("median", command)


if __name__ == "__main__":
    unittest.main()
