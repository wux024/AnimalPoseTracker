import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from animalposetracker.nn.transformer import VisionTransformer, resize_pos_embed
from animalposetracker.nn.spec import parse as parse_model_spec
from animalposetracker.training.checkpoint import load_model_weights
from animalposetracker.training.cli import _resolve_project_pretrained_path
from animalposetracker.training.config import TrainingConfig
from animalposetracker.training.engine import build_optimizer
from animalposetracker.evaluation.metrics import (
    build_coco_ground_truth,
    evaluate_coco_keypoints,
    _coco_keypoint_api,
)
from animalposetracker.evaluation.animalrtpose import PoseDetectionValidator
from animalposetracker.data.pose import pose_collate
from animalposetracker.training.profiles import (
    TRAINING_PROFILES,
    apply_animalvitpose_defaults,
    configure_animalvitpose_model,
    flatten_project_training_config,
)
from animalposetracker.project.api import (
    AnimalPoseTrackerProject,
    _download_google_drive_checkpoint,
)
from animalposetracker.export.cli import _resolve_export_format
from animalposetracker.data.simcc import SimCCLabel
from animalposetracker.training.simcc import SimCCKLLoss
from animalposetracker.evaluation.topdown import SimCCPoseValidator
from animalposetracker.data.topdown import (
    TopDownPoseDataset,
    build_topdown_dataloaders,
)


class SimCCTrainingTests(unittest.TestCase):
    def test_animalrtpose_checkpoint_urls_live_with_model_config_and_download(self):
        expected_ids = {
            "n": "1frHtyJKbyAZUr4uGpbTSje4L0hrkeqgz",
            "s": "19XeuqZRvF9IBoBbPPkDrjew3Pw9MdBLJ",
            "m": "1BRlOoZnVHyf_P_UJJWhhw9udF3HsNBT6",
            "l": "1CeQTPO4Mf-uG3jwkD5rnzttn6yFZoHjK",
            "x": "1ubZL6AVGuLb8616Y6lySQKjNNbV85qUL",
        }
        model_yaml = (
            Path(__file__).parent.parent
            / "animalposetracker"
            / "cfg"
            / "models"
            / "animalrtpose.yaml"
        )
        model_values = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
        expected_urls = {
            scale: f"https://drive.google.com/uc?export=download&id={file_id}"
            for scale, file_id in expected_ids.items()
        }
        self.assertEqual(model_values["pretrained_urls"], expected_urls)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project = AnimalPoseTrackerProject(
                local_path=temporary_directory,
                project_name="rtpose-pretrained-test",
                worker="test",
                model_type="AnimalRTPose",
                model_scale="N",
                keypoints=3,
                classes=1,
            )
            project.create_new_project()
            # Simulate a project created before URLs were colocated with its model yaml.
            project.model_config.pop("pretrained_urls", None)
            project.other_config["pretrained"] = True
            downloaded = project.project_path / "pretrained" / "animalrtpose-n.pt"

            def write_checkpoint(_url, destination):
                Path(destination).write_bytes(b"PK\x03\x04checkpoint-bytes")

            with patch(
                "animalposetracker.project.api._download_google_drive_checkpoint",
                side_effect=write_checkpoint,
            ) as download:
                project._detect_pretrained()

            download.assert_called_once_with(expected_urls["n"], downloaded)
            self.assertEqual(downloaded.read_bytes(), b"PK\x03\x04checkpoint-bytes")
            self.assertEqual(project.other_config["pretrained"], str(downloaded))

    def test_google_drive_checkpoint_download_follows_confirmation_form(self):
        url = "https://drive.google.com/uc?export=download&id=file-id"
        action = "https://drive.usercontent.google.com/download"
        confirmation_html = (
            '<form id="download-form" action="https://drive.usercontent.google.com/download">'
            '<input type="hidden" name="id" value="file-id">'
            '<input type="hidden" name="export" value="download">'
            '<input type="hidden" name="confirm" value="t">'
            '<input type="hidden" name="uuid" value="confirmation-id">'
            "</form>"
        )
        confirmation_response = Mock()
        confirmation_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        confirmation_response.text = confirmation_html
        file_response = Mock()
        file_response.headers = {"Content-Type": "application/octet-stream"}
        file_response.iter_content.return_value = [b"PK\x03\x04checkpoint-bytes"]
        session = Mock()
        session.get.side_effect = [confirmation_response, file_response]

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "animalrtpose-m.pt"
            with patch("requests.Session", return_value=session):
                _download_google_drive_checkpoint(url, destination)

            self.assertEqual(destination.read_bytes(), b"PK\x03\x04checkpoint-bytes")

        self.assertEqual(
            session.get.call_args_list,
            [
                call(url, stream=True, timeout=30),
                call(
                    action,
                    params={
                        "id": "file-id",
                        "export": "download",
                        "confirm": "t",
                        "uuid": "confirmation-id",
                    },
                    stream=True,
                    timeout=60,
                ),
            ],
        )
        confirmation_response.raise_for_status.assert_called_once_with()
        file_response.raise_for_status.assert_called_once_with()

    def test_export_format_requires_an_explicit_target(self):
        self.assertEqual(_resolve_export_format(" ONNX ", None), "onnx")
        self.assertEqual(_resolve_export_format(None, "torchscript"), "torchscript")
        with self.assertRaisesRegex(ValueError, "No export format selected"):
            _resolve_export_format(None, None)

    def test_equal_model_defaults_are_shared_and_different_values_stay_overrides(self):
        profiles = TRAINING_PROFILES["models"]
        sections = ("training", "loss", "augmentation", "validation")
        for section in sections:
            rtp = profiles["AnimalRTPose"].get(section, {}) or {}
            vit = profiles["AnimalViTPose"].get(section, {}) or {}
            equal_duplicates = {
                key for key in rtp.keys() & vit.keys() if rtp[key] == vit[key]
            }
            self.assertEqual(equal_duplicates, set(), f"duplicate defaults in {section}")

        shared_training = TRAINING_PROFILES["shared"]["training"]
        for key in (
            "lr_gamma",
            "single_cls",
            "multi_scale",
            "freeze",
            "best_metric",
            "best_metric_mode",
        ):
            self.assertIn(key, shared_training)
        self.assertEqual(profiles["AnimalRTPose"]["training"]["batch_size"], 16)
        self.assertEqual(profiles["AnimalViTPose"]["training"]["batch_size"], 64)

    def test_gaussian_labels_follow_simcc_grid_and_mask_visibility(self):
        codec = SimCCLabel((64, 64), sigma=3.0, split_ratio=2.0, normalize=True)
        label_x, label_y, weights = codec.encode(
            np.asarray([[20.0, 30.0], [-200.0, 10.0], [40.0, 40.0]], dtype=np.float32),
            np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        )

        self.assertEqual(label_x.shape, (3, 128))
        self.assertEqual(label_y.shape, (3, 128))
        self.assertEqual(int(label_x[0].argmax()), 40)
        self.assertEqual(int(label_y[0].argmax()), 60)
        self.assertAlmostEqual(float(label_x[0].sum()), 1.0, places=4)
        self.assertEqual(weights.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(float(label_x[1].sum()), 0.0)
        self.assertEqual(float(label_x[2].sum()), 0.0)

    def test_kl_loss_sums_axes_and_propagates_gradients(self):
        keypoints = 2
        pred_x = torch.randn(2, keypoints, 16, requires_grad=True)
        pred_y = torch.randn(2, keypoints, 16, requires_grad=True)
        target_x = torch.softmax(torch.randn(2, keypoints, 16), dim=-1)
        target_y = torch.softmax(torch.randn(2, keypoints, 16), dim=-1)
        weights = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        result = SimCCKLLoss(keypoints)(
            (pred_x, pred_y),
            {"simcc_x": target_x, "simcc_y": target_y, "keypoint_weights": weights},
        )

        expected = 0.0
        for prediction, target in ((pred_x, target_x), (pred_y, target_y)):
            per_keypoint = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(prediction.reshape(-1, 16), dim=1),
                target.reshape(-1, 16),
                reduction="none",
            ).mean(dim=1)
            expected = expected + (per_keypoint * weights.reshape(-1)).sum() / keypoints
        self.assertTrue(torch.allclose(result["loss"], expected))
        self.assertTrue(SimCCKLLoss.loss_is_batch_sum)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(pred_x.grad).all())
        self.assertTrue(torch.isfinite(pred_y.grad).all())

    def test_negative_vit_output_index_selects_last_layer(self):
        model = VisionTransformer(
            arch={
                "embed_dims": 32,
                "num_layers": 2,
                "num_heads": 4,
                "feedforward_channels": 64,
            },
            img_size=32,
            patch_size=16,
            out_indices=-1,
            drop_path_rate=0.1,
        )
        features = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(features.shape), (1, 32, 2, 2))
        self.assertEqual(model.out_indices, [1])

    def test_mae_position_embedding_drops_cls_token_and_resizes(self):
        resized = resize_pos_embed(torch.randn(1, 197, 32), 256, num_extra_tokens=0)
        self.assertEqual(tuple(resized.shape), (1, 256, 32))

    def test_mmpose_checkpoint_prefix_and_cls_position_are_adapted(self):
        class TinyGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                backbone = torch.nn.Module()
                backbone.num_extra_tokens = 0
                backbone.pos_embed = torch.nn.Parameter(torch.zeros(1, 4, 2))
                self.model = torch.nn.ModuleList([backbone])

        with tempfile.TemporaryDirectory() as temporary_directory:
            model = TinyGraph()
            reports = []
            for filename, key in (("mmpose.pth", "backbone.pos_embed"), ("mae.pth", "pos_embed")):
                checkpoint = Path(temporary_directory) / filename
                torch.save({"state_dict": {key: torch.ones(1, 197, 2)}}, checkpoint)
                reports.append(load_model_weights(checkpoint, model, strict=False))

        for report in reports:
            self.assertEqual(report["loaded_tensors"], 1)
            self.assertEqual(report["shape_mismatches"], [])
        self.assertEqual(tuple(model.model[0].pos_embed.shape), (1, 4, 2))

    def test_remote_pretrained_checkpoint_is_cached_in_project_once(self):
        url = "https://example.invalid/mae_pretrain_vit_small.pth"
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "animalposetracker.training.checkpoint.urlopen",
                return_value=io.BytesIO(b"checkpoint-data"),
            ) as open_url:
                resolved = _resolve_project_pretrained_path(url, Path(temporary_directory))
            self.assertEqual(resolved.read_bytes(), b"checkpoint-data")
            self.assertEqual(resolved.parent, Path(temporary_directory) / "pretrained")
            with patch("animalposetracker.training.checkpoint.urlopen", side_effect=AssertionError):
                cached = _resolve_project_pretrained_path(url, Path(temporary_directory))
            self.assertEqual(cached, resolved)
            self.assertEqual(open_url.call_count, 1)

    def test_user_cached_pretrained_checkpoint_is_copied_into_project(self):
        url = "https://example.invalid/mae_pretrain_vit_small.pth"
        with tempfile.TemporaryDirectory() as temporary_directory:
            home = Path(temporary_directory) / "home"
            shared_cache = home / ".cache" / "animalposetracker" / "pretrained"
            shared_cache.mkdir(parents=True)
            (shared_cache / "mae_pretrain_vit_small.pth").write_bytes(b"cached-weights")
            project_dir = Path(temporary_directory) / "project"

            with patch("animalposetracker.training.cli.Path.home", return_value=home):
                with patch(
                    "animalposetracker.training.checkpoint.urlopen",
                    side_effect=AssertionError("existing cache should avoid a network request"),
                ):
                    resolved = _resolve_project_pretrained_path(url, project_dir)

            self.assertEqual(
                resolved,
                project_dir / "pretrained" / "mae_pretrain_vit_small.pth",
            )
            self.assertEqual(resolved.read_bytes(), b"cached-weights")

    def test_mmpose_defaults_preserve_explicit_experiment_settings(self):
        values, applied = apply_animalvitpose_defaults({
            "epochs": 2,
            "batch": 3,
            "imgsz": 256,
            "optimizer": "AdamW",
            "lr0": 1e-5,
            "weight_decay": 0.01,
            "validation_interval": 1,
            "best_metric": "coco/AP",
            "amp": False,
            "pretrained": False,
        })
        self.assertEqual(values["epochs"], 2)
        self.assertEqual(values["batch"], 3)
        self.assertEqual(values["validation_interval"], 1)
        self.assertEqual(values["best_metric"], "coco/AP")
        self.assertEqual(values["seed"], 21)
        self.assertEqual(values["log_interval"], 50)
        self.assertEqual(values["warmup_start_factor"], 0.001)
        self.assertEqual(values["gradient_accumulation_steps"], 1)
        self.assertIn("layer_decay_rate", applied)

    def test_generic_project_defaults_switch_to_the_animalvitpose_recipe(self):
        values, _applied = apply_animalvitpose_defaults({
            "epochs": 1000,
            "batch": 16,
            "imgsz": 640,
            "workers": 8,
            "optimizer": "auto",
            "lr0": 0.01,
            "weight_decay": 5e-4,
            "amp": True,
            "patience": 300,
            "pretrained": True,
        })
        self.assertEqual(values["epochs"], 500)
        self.assertEqual(values["batch_size"], 64)
        self.assertEqual(values["validation_batch_size"], 32)
        self.assertEqual(values["image_size"], 256)
        self.assertEqual(values["optimizer"], "AdamW")
        self.assertEqual(values["learning_rate"], 5e-4)
        self.assertEqual(values["weight_decay"], 0.1)
        self.assertEqual(values["validation_interval"], 10)
        self.assertEqual(values["best_metric"], "coco/AP")
        self.assertEqual(values["ema"], False)
        self.assertEqual(values["amp"], False)
        self.assertEqual(values["early_stopping_patience"], 0)
        self.assertEqual(values["gradient_accumulation_steps"], 1)
        self.assertIn("mae_pretrain_vit_small_20230913.pth", values["pretrained_weights"])
        typed = TrainingConfig.from_mapping(values)
        self.assertEqual(typed.pretrained_weights, values["pretrained_weights"])

    def test_vit_layer_decay_groups_match_mmpose_layer_order(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pos_embed = torch.nn.Parameter(torch.zeros(1, 4, 2))
                self.patch_embed = torch.nn.Linear(2, 2)
                self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])

        class TinyGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleList([TinyBackbone(), torch.nn.Linear(2, 2)])

        config = TrainingConfig.from_mapping({
            "optimizer": "AdamW",
            "learning_rate": 1e-2,
            "weight_decay": 0.1,
            "layer_decay_rate": 0.8,
        })
        optimizer = build_optimizer(TinyGraph(), config)
        rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(rates["layer_0_decay"], 1e-2 * (0.8 ** 3))
        self.assertAlmostEqual(rates["layer_1_decay"], 1e-2 * (0.8 ** 2))
        self.assertAlmostEqual(rates["layer_2_decay"], 1e-2 * 0.8)
        self.assertAlmostEqual(rates["layer_3_decay"], 1e-2)
        no_decay = next(group for group in optimizer.param_groups if group["group_name"] == "layer_3_no_decay")
        self.assertEqual(no_decay["weight_decay"], 0.0)

    def test_mmpose_variant_and_resolution_update_the_single_model_graph(self):
        model_spec = {
            "backbone": [[-1, 1, "ViT", [384, 12, 12, 1536, 16, 256, {"padding": 2}, {}]]],
            "head": [[[-1], 1, "SimCCHead", [17, [256, 256], [16, 16], 2.0, [256], [4]]]],
        }
        variant = configure_animalvitpose_model(model_spec, "large", 384)
        self.assertEqual(variant, "large")
        self.assertEqual(model_spec["backbone"][0][3][:6], [1024, 24, 16, 4096, 16, 384])
        self.assertEqual(model_spec["backbone"][0][3][7]["drop_path_rate"], 0.5)
        self.assertEqual(model_spec["head"][0][3][1], [384, 384])
        self.assertEqual(model_spec["head"][0][3][2], [24, 24])
        profile, _applied = apply_animalvitpose_defaults({}, model_scale="huge")
        self.assertEqual(profile["layer_decay_rate"], 0.85)

    def test_coco_instances_are_cropped_and_encoded_as_simcc(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images" / "train"
            image_dir.mkdir(parents=True)
            Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8)).save(image_dir / "one.png")
            annotation_dir = root / "annotations"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "images": [{"id": 1, "file_name": "images/train/one.png", "width": 64, "height": 64}],
                "annotations": [{
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [8, 8, 48, 48],
                    "area": 2304,
                    "keypoints": [20, 20, 2, 44, 44, 2],
                    "num_keypoints": 2,
                    "iscrowd": 0,
                }],
                "categories": [{
                    "id": 1,
                    "name": "animal",
                    "keypoints": ["left", "right"],
                    "skeleton": [[1, 2]],
                }],
            }
            train_annotation = annotation_dir / "train.json"
            train_annotation.write_text(json.dumps(payload), encoding="utf-8")
            (annotation_dir / "val.json").write_text(json.dumps(payload), encoding="utf-8")
            config_path = root / "dataset.yaml"
            config_path.write_text(
                "path: .\n"
                "annotation_format: coco\n"
                "train: images/train\n"
                "val: images/train\n"
                "train_annotations: annotations/train.json\n"
                "val_annotations: annotations/val.json\n"
                "kpt_shape: [2, 3]\n"
                "kpt_oks_sigmas: [0.08, 0.14]\n"
                "flip_idx: [0, 1]\n"
                "names: {0: animal}\n",
                encoding="utf-8",
            )

            dataset = TopDownPoseDataset(
                config_path,
                "val",
                input_size=(32, 32),
                sigma=3.0,
                cache=False,
            )
            sample = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(sample["images"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["targets"]["simcc_x"].shape), (2, 64))
            self.assertEqual(tuple(sample["targets"]["simcc_y"].shape), (2, 64))
            self.assertEqual(sample["targets"]["image_id"], 1)
            self.assertTrue(torch.equal(sample["targets"]["keypoint_weights"], torch.ones(2)))
            self.assertTrue(np.allclose(dataset.kpt_oks_sigmas, [0.08, 0.14]))

    def test_coco_keypoint_ap_uses_dataset_sigma_vector(self):
        COCO, _COCOeval = _coco_keypoint_api()
        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {},
            "images": [{"id": 1, "file_name": "mouse.png", "width": 20, "height": 20}],
            "categories": [{"id": 1, "name": "mouse", "keypoints": ["nose", "tail"], "skeleton": []}],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": [5, 5, 2, 10, 10, 2],
                "num_keypoints": 2,
                "bbox": [0, 0, 20, 20],
                "area": 100,
                "iscrowd": 0,
            }],
        }
        coco_gt.createIndex()
        detections = [{
            "image_id": 1,
            "category_id": 1,
            "keypoints": [10, 5, 0.9, 10, 10, 0.9],
            "score": 0.9,
        }]

        strict = evaluate_coco_keypoints(coco_gt, detections, [0.001, 0.001], [1], [1])
        tolerant = evaluate_coco_keypoints(coco_gt, detections, [1.0, 1.0], [1], [1])
        self.assertLess(strict["coco/AP"], tolerant["coco/AP"])

    def test_yolo_validation_labels_are_converted_to_coco_keypoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8)).save(image_dir / "mouse.png")
            (label_dir / "mouse.txt").write_text(
                "0 0.5 0.5 0.5 0.5 0.25 0.25 2 0.75 0.75 2\n",
                encoding="utf-8",
            )
            config_path = root / "dataset.yaml"
            config_path.write_text(
                "path: .\n"
                "train: images/val\n"
                "val: images/val\n"
                "kpt_shape: [2, 3]\n"
                "kpt_oks_sigmas: [0.08, 0.14]\n"
                "keypoint_names: [nose, tail]\n"
                "names: {0: mouse}\n",
                encoding="utf-8",
            )
            from animalposetracker.data.pose import PoseTextDataset

            dataset = PoseTextDataset(config_path, "val", image_size=32)
            coco_gt, image_ids, category_ids = build_coco_ground_truth(dataset)
            annotation = coco_gt.dataset["annotations"][0]
            self.assertEqual(annotation["keypoints"], [16.0, 16.0, 2.0, 48.0, 48.0, 2.0])
            self.assertEqual(image_ids[str((image_dir / "mouse.png").resolve())], 1)
            self.assertEqual(category_ids, {0: 1})
            result = evaluate_coco_keypoints(
                coco_gt,
                [{
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [16, 16, 0.9, 48, 48, 0.9],
                    "score": 0.9,
                }],
                dataset.kpt_oks_sigmas,
                image_ids=[1],
                category_ids=[1],
            )
            self.assertAlmostEqual(result["coco/AP"], 1.0)

            class FakeCriterion(torch.nn.Module):
                num_classes = 1
                num_keypoints = 2
                keypoint_dimensions = 3

                def __init__(self):
                    super().__init__()
                    self.keypoint_loss = torch.nn.Module()
                    self.keypoint_loss.register_buffer(
                        "kpt_oks_sigmas", torch.tensor(dataset.kpt_oks_sigmas)
                    )

                def forward(self, _raw_predictions, _targets):
                    return {"metrics": {"val_loss": torch.tensor(0.1)}}

            class FakeModel(torch.nn.Module):
                def forward(self, images):
                    decoded = images.new_tensor(
                        [[16], [16], [16], [16], [0.99], [8], [8], [0.9], [24], [24], [0.9]]
                    ).reshape(1, 11, 1)
                    return decoded, None

            loader = DataLoader(dataset, batch_size=1, collate_fn=pose_collate)
            validator = PoseDetectionValidator(
                FakeCriterion(), validation_dataset=dataset, max_detections=20
            )
            validation_result = validator(FakeModel(), loader, torch.device("cpu"))
            self.assertAlmostEqual(validation_result["coco/AP"], 1.0)
            self.assertAlmostEqual(validation_result["fitness"], validation_result["coco/AP"])
            self.assertAlmostEqual(validation_result["PCK"], 1.0)
            self.assertAlmostEqual(validation_result["AUC"], 0.95)
            self.assertAlmostEqual(validation_result["EPE"], 0.0)

            topdown_train, topdown_val, topdown_metadata = build_topdown_dataloaders(
                config_path,
                input_size=(32, 32),
                batch_size=1,
                validation_batch_size=1,
                num_workers=0,
            )
            self.assertIsNone(topdown_metadata["annotation_path"])
            target_sample = topdown_val.dataset[0]["targets"]

            class ExactSimCCModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("target_x", target_sample["simcc_x"] * 100.0)
                    self.register_buffer("target_y", target_sample["simcc_y"] * 100.0)

                def forward(self, images):
                    batch_size = images.shape[0]
                    return (
                        self.target_x.unsqueeze(0).expand(batch_size, -1, -1),
                        self.target_y.unsqueeze(0).expand(batch_size, -1, -1),
                    )

            simcc_validator = SimCCPoseValidator(
                input_size=(32, 32),
                split_ratio=2.0,
                flip_indices=None,
                kpt_oks_sigmas=topdown_metadata["kpt_oks_sigmas"],
                validation_dataset=topdown_val.dataset,
            )
            simcc_result = simcc_validator(
                ExactSimCCModel(), topdown_val, torch.device("cpu")
            )
            self.assertIn("coco/AP", simcc_result)
            self.assertIn("PCK", simcc_result)
            self.assertIn("AUC", simcc_result)
            self.assertIn("EPE", simcc_result)

    def test_noncanonical_oks_sigma_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images" / "val"
            image_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "one.png")
            config_path = root / "dataset.yaml"
            for field in ("oks_sigmas", "sigmas"):
                config_path.write_text(
                    "path: .\n"
                    "val: images/val\n"
                    "kpt_shape: [1, 3]\n"
                    f"{field}: [0.1]\n"
                    "names: {0: animal}\n",
                    encoding="utf-8",
                )
                from animalposetracker.data.pose import PoseTextDataset

                with self.assertRaisesRegex(ValueError, "rename the vector to kpt_oks_sigmas"):
                    PoseTextDataset(config_path, "val", image_size=16)

    def test_animalvitpose_project_creation_writes_four_config_types_and_standard_profile(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = AnimalPoseTrackerProject(
                local_path=temporary_directory,
                project_name="vit-project",
                worker="test",
                model_type="AnimalViTPose",
                model_scale="S",
                keypoints=12,
                classes=3,
                keypoints_name=[f"point_{index}" for index in range(12)],
                skeleton=[[0, 1]],
            )
            project.create_new_project()
            project_root = project.project_path
            config_dir = project_root / "configs"
            dataset_config = yaml.safe_load(
                (config_dir / "dataset.yaml").read_text(encoding="utf-8")
            )
            model_config = yaml.safe_load(
                (config_dir / "model.yaml").read_text(encoding="utf-8")
            )
            training_config = yaml.safe_load(
                (config_dir / "other.yaml").read_text(encoding="utf-8")
            )

            self.assertTrue((project_root / "project.yaml").is_file())
            self.assertTrue((config_dir / "dataset.yaml").is_file())
            self.assertTrue((config_dir / "model.yaml").is_file())
            self.assertTrue((config_dir / "other.yaml").is_file())
            self.assertEqual(
                {path.name for path in config_dir.glob("*.yaml")},
                {"dataset.yaml", "model.yaml", "other.yaml"},
            )
            project_metadata = yaml.safe_load(
                (project_root / "project.yaml").read_text(encoding="utf-8")
            )
            self.assertNotIn("kpt_oks_sigmas", project_metadata)
            self.assertNotIn("task", training_config)
            self.assertTrue(training_config["model"].endswith("model.yaml"))
            profile = training_config["training"]
            self.assertNotIn("active_model", profile)
            self.assertNotIn("model_scale", profile)
            self.assertNotIn("copy_paste", profile["shared"])
            self.assertEqual(profile["shared"]["training"]["lr_gamma"], 0.1)
            self.assertNotIn("lr_gamma", profile["models"]["AnimalViTPose"]["training"])
            self.assertEqual(profile["models"]["AnimalViTPose"]["training"]["batch_size"], 64)
            self.assertNotIn("model_scales", profile["models"]["AnimalViTPose"])
            self.assertNotIn("preprocessing", profile["models"]["AnimalViTPose"])
            self.assertNotIn("simcc", profile["models"]["AnimalViTPose"])
            self.assertEqual(profile["models"]["AnimalViTPose"]["loss"]["label_sigma"], 6.0)
            self.assertEqual(
                profile["models"]["AnimalViTPose"]["training"]["layer_decay_rate_by_scale"]["small"],
                0.8,
            )
            self.assertEqual(model_config["model_variants"]["small"]["num_layers"], 12)
            self.assertEqual(model_config["nc"], 1)
            self.assertEqual(model_config["head"][-1][3][0], "num_keypoints")
            parsed_model = parse_model_spec(model_config)
            self.assertEqual(parsed_model.kpt_shape, (12, 3))
            self.assertEqual(parsed_model.head_layer().args[0]["out_channels"], 12)
            self.assertEqual(model_config["preprocessing"]["bbox_padding"], 1.25)
            configured_model = yaml.safe_load(
                (config_dir / "model.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(configure_animalvitpose_model(configured_model, "base", 256), "base")
            self.assertEqual(configured_model["backbone"][0][3][:4], [768, 12, 12, 3072])
            materialized = flatten_project_training_config(
                training_config,
                model_type="AnimalViTPose",
            )
            self.assertEqual(materialized["batch"], 64)
            self.assertEqual(materialized["imgsz"], 256)
            self.assertEqual(project.other_config["batch"], 64)
            self.assertEqual(project.other_config["imgsz"], 256)
            self.assertEqual(project.other_config["optimizer"], "AdamW")
            self.assertEqual(project.other_config["label_sigma"], 6.0)
            self.assertNotIn("box", project.other_config)
            self.assertFalse(project.other_config["ema"])
            self.assertNotIn("batch", training_config)
            self.assertIn("kpt_oks_sigmas", dataset_config)
            self.assertNotIn("oks_sigmas", dataset_config)

            # Existing projects may still carry the old YOLO task selector.
            training_config["task"] = "pose"
            (config_dir / "other.yaml").write_text(
                yaml.safe_dump(training_config, sort_keys=False),
                encoding="utf-8",
            )
            project._load_config_from_file("other", config_dir / "other.yaml")
            self.assertNotIn("task", project.other_config)

            project.update_config("other", {"batch": 7, "lr_gamma": 0.2, "seed": 24})
            project.save_configs("other")
            saved_other = yaml.safe_load(
                (config_dir / "other.yaml").read_text(encoding="utf-8")
            )
            self.assertNotIn("task", saved_other)
            self.assertEqual(saved_other["training"]["shared"]["training"]["lr_gamma"], 0.2)
            self.assertEqual(saved_other["training"]["shared"]["runtime"]["seed"], 0)
            self.assertEqual(
                saved_other["training"]["models"]["AnimalViTPose"]["training"]["seed"],
                24,
            )
            self.assertNotIn(
                "lr_gamma",
                saved_other["training"]["models"]["AnimalViTPose"]["training"],
            )
            self.assertEqual(
                saved_other["training"]["models"]["AnimalViTPose"]["training"]["batch_size"],
                7,
            )

    def test_project_evaluation_uses_custom_validator_cli(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = AnimalPoseTrackerProject(
                local_path=temporary_directory,
                project_name="eval-project",
                worker="test",
                model_type="AnimalRTPose",
                model_scale="N",
                keypoints=3,
                classes=1,
            )
            project.create_new_project()
            checkpoint = project.project_path / "runs" / "train" / "weights" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            with patch.object(project, "_execute_command") as execute:
                project.evaluate()

            command = execute.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertIn("animalposetracker.training.cli", command)
            self.assertIn("--validate-only", command)
            self.assertIn(str(checkpoint.resolve()), command)
            self.assertNotIn("yolo", command)

            project.other_config["model"] = str(checkpoint)
            with patch.object(project, "_execute_command") as execute:
                project.predict(inference_source="input.png")
            predict_command = execute.call_args.args[0]
            self.assertIn("animalposetracker.prediction.cli", predict_command)
            self.assertIn("--source", predict_command)
            self.assertIn("input.png", predict_command)
            self.assertNotIn("yolo", predict_command)

            with patch.object(project, "_execute_command") as execute:
                project.predict()
            dataset_predict_command = execute.call_args.args[0]
            self.assertIn("animalposetracker.prediction.cli", dataset_predict_command)
            self.assertNotIn("--source", dataset_predict_command)

            with patch.object(project, "_execute_command") as execute:
                project.export(format="onnx")
            export_command = execute.call_args.args[0]
            self.assertIn("animalposetracker.export.cli", export_command)
            self.assertIn("--weights", export_command)
            self.assertIn("--format", export_command)
            self.assertIn("onnx", export_command)
            self.assertNotIn("yolo", export_command)

            saved_other = yaml.safe_load(
                (project.project_path / "configs" / "other.yaml").read_text(encoding="utf-8")
            )
            self.assertTrue(saved_other["model"].endswith("model.yaml"))
            self.assertIsNone(saved_other["application"]["format"])

    def test_project_prediction_resolves_exported_model_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = AnimalPoseTrackerProject(
                local_path=temporary_directory,
                project_name="exported-model-project",
                worker="test",
                model_type="AnimalRTPose",
                model_scale="N",
                keypoints=3,
                classes=1,
            )
            project.create_new_project()
            artifact = project.project_path / "runs" / "export" / "pose_openvino_model"
            artifact.mkdir(parents=True)
            project.other_config["model"] = str(artifact)
            self.assertEqual(project._resolve_project_prediction_model(), artifact.resolve())

    def test_project_predict_keeps_dataset_mode_separate_from_live_inference_engine(self):
        source = (Path(__file__).parent.parent / "animalposetracker" / "prediction" / "cli.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("animalposetracker.inference.inferencer", source)
        from animalposetracker.artifacts import EXPORT_FORMATS, PREDICT_FORMATS, detect_artifact_format

        self.assertEqual(len(EXPORT_FORMATS), 15)
        self.assertEqual(len(PREDICT_FORMATS), 16)
        self.assertIn("onnx", EXPORT_FORMATS)
        self.assertIn("rknn", EXPORT_FORMATS)
        self.assertIn("saved_model", PREDICT_FORMATS)
        self.assertIn("triton", PREDICT_FORMATS)
        self.assertEqual(detect_artifact_format("grpc://localhost:8001/animal_pose"), "triton")

    def test_topdown_prediction_uses_dataset_or_detector_box_provider(self):
        from animalposetracker.prediction.cli import (
            DetectorBoxProvider,
            _predict_topdown_image,
        )

        class FixedSimCC(torch.nn.Module):
            def forward(self, images):
                pred_x = torch.zeros((images.shape[0], 2, 64), dtype=images.dtype)
                pred_y = torch.zeros((images.shape[0], 2, 64), dtype=images.dtype)
                pred_x[:, 0, 20] = 5.0
                pred_x[:, 1, 44] = 4.0
                pred_y[:, 0, 22] = 5.0
                pred_y[:, 1, 42] = 4.0
                return pred_x, pred_y

        provider = DetectorBoxProvider(
            lambda _image: [{"bbox_xyxy": [8, 8, 56, 56], "class_id": 0}]
        )
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        context = {
            "device": torch.device("cpu"),
            "topdown_input_size": (32, 32),
            "simcc_split_ratio": 2.0,
            "preprocessing": {
                "bbox_padding": 1.25,
                "pixel_mean": [123.675, 116.28, 103.53],
                "pixel_std": [58.395, 57.12, 57.375],
            },
            "class_names": ["mouse"],
            "skeleton": [[0, 1]],
        }
        settings = {
            "show_boxes": True,
            "show_labels": True,
            "show_keypoints": True,
            "show_skeletons": True,
            "keypoint_confidence": 0.1,
            "point_radius": 2,
            "line_width": 1,
        }
        rendered, records = _predict_topdown_image(
            image,
            Path("mouse.png"),
            provider.boxes_for_image(image, Path("mouse.png")),
            FixedSimCC(),
            context,
            settings,
        )
        self.assertEqual(rendered.shape, image.shape)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["class_name"], "mouse")
        self.assertEqual(len(records[0]["keypoints"]), 2)

    def test_animalrtpose_default_dataset_prediction_does_not_require_test_labels(self):
        from animalposetracker.prediction.cli import _resolve_image_split

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            test_images = root / "images" / "test"
            test_images.mkdir(parents=True)
            image_path = test_images / "unlabeled.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
            data_path = root / "dataset.yaml"
            data_path.write_text(
                "path: .\n"
                "annotation_format: coco\n"
                "test: images/test\n"
                "val: images/test\n"
                "kpt_shape: [1, 3]\n"
                "names: {0: mouse}\n",
                encoding="utf-8",
            )
            split = _resolve_image_split(data_path, "auto")
            self.assertEqual(split.split, "test")
            self.assertEqual(split.image_paths, [image_path.resolve()])

    def test_exported_animalrtpose_output_uses_native_pose_postprocessing(self):
        from animalposetracker.prediction.cli import _predict_frame
        from animalposetracker.training.config import TrainingConfig

        prediction = torch.tensor(
            [[16.0], [16.0], [16.0], [16.0], [0.99], [8.0], [8.0], [0.9], [24.0], [24.0], [0.9]],
            dtype=torch.float32,
        ).reshape(1, 11, 1)

        class FakeExportedBackend:
            def __call__(self, _batch):
                return [prediction.numpy()]

        context = {
            "device": torch.device("cpu"),
            "model_nc": 1,
            "kpt_shape": (2, 3),
            "class_names": ["mouse"],
            "skeleton": [[0, 1]],
        }
        settings = {
            "confidence": 0.1,
            "keypoint_confidence": 0.1,
            "iou": 0.7,
            "max_detections": 10,
            "agnostic_nms": False,
            "show_boxes": True,
            "show_labels": True,
            "show_keypoints": True,
            "show_skeletons": True,
            "point_radius": 2,
            "line_width": 1,
        }
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        rendered, records = _predict_frame(
            image,
            FakeExportedBackend(),
            TrainingConfig.from_mapping({"image_size": 32}),
            context,
            settings,
            external_backend=True,
        )
        self.assertEqual(rendered.shape, image.shape)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["class_name"], "mouse")

    def test_animalvitpose_project_prediction_runs_from_annotated_dataset_split(self):
        from animalposetracker.nn import build_model
        from animalposetracker.export.cli import run as run_export
        from animalposetracker.prediction.cli import run as run_prediction
        from animalposetracker.project.model_context import (
            configure_project_model_spec,
            load_project_context,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project = AnimalPoseTrackerProject(
                local_path=temporary_directory,
                project_name="vit-predict-project",
                worker="test",
                model_type="AnimalViTPose",
                model_scale="small",
                keypoints=2,
                classes=1,
                keypoints_name=["nose", "tail"],
                skeleton=[[0, 1]],
                classes_name=["mouse"],
            )
            project.create_new_project()
            root = project.project_path
            dataset_root = root / "datasets"
            for split in ("train", "val"):
                image_dir = dataset_root / "images" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8)).save(image_dir / "one.png")
            annotation_dir = dataset_root / "annotations"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            annotation = {
                "images": [{
                    "id": 1,
                    "file_name": "images/val/one.png",
                    "width": 64,
                    "height": 64,
                }],
                "annotations": [{
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [8, 8, 48, 48],
                    "area": 2304,
                    "keypoints": [20, 20, 2, 44, 44, 2],
                    "num_keypoints": 2,
                    "iscrowd": 0,
                }],
                "categories": [{
                    "id": 1,
                    "name": "mouse",
                    "keypoints": ["nose", "tail"],
                    "skeleton": [[1, 2]],
                }],
            }
            validation_annotations = annotation_dir / "val.json"
            validation_annotations.write_text(json.dumps(annotation), encoding="utf-8")
            (annotation_dir / "train.json").write_text(json.dumps({
                **annotation,
                "images": [{
                    "id": 1,
                    "file_name": "images/train/one.png",
                    "width": 64,
                    "height": 64,
                }],
            }), encoding="utf-8")
            dataset_values = {
                "path": str(dataset_root),
                "annotation_format": "coco",
                "train": "images/train",
                "val": "images/val",
                "train_annotations": "annotations/train.json",
                "val_annotations": "annotations/val.json",
                "kpt_shape": [2, 3],
                "kpt_oks_sigmas": [0.08, 0.14],
                "flip_idx": [0, 1],
                "names": {0: "mouse"},
                "skeleton": [[0, 1]],
            }
            dataset_path = root / "configs" / "dataset.yaml"
            dataset_path.write_text(yaml.safe_dump(dataset_values, sort_keys=False), encoding="utf-8")

            model_values = {
                "nc": 1,
                "kpt_shape": [2, 3],
                "backbone": [[-1, 1, "ViT", [32, 2, 4, 64, 16, 32, {"padding": 0}, {"drop_path_rate": 0.0}]]],
                "head": [[[-1], 1, "SimCCHead", ["num_keypoints", [32, 32], [2, 2], 2.0, [8], [4]]]],
                "model_variants": {
                    "small": {
                        "embed_dims": 32,
                        "num_layers": 2,
                        "num_heads": 4,
                        "feedforward_channels": 64,
                        "drop_path_rate": 0.0,
                    },
                },
                "simcc": {"split_ratio": 2.0},
                "preprocessing": {
                    "pixel_mean": [123.675, 116.28, 103.53],
                    "pixel_std": [58.395, 57.12, 57.375],
                    "bbox_padding": 1.25,
                },
            }
            model_path = root / "configs" / "model.yaml"
            model_path.write_text(yaml.safe_dump(model_values, sort_keys=False), encoding="utf-8")
            project.other_config["imgsz"] = 32
            project.save_configs("other")

            config, context = load_project_context(root / "configs" / "other.yaml")
            model_spec, _head_name, scale = configure_project_model_spec(config, context)
            model = build_model(model_spec, scale=scale)
            checkpoint = root / "test-weights.pt"
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            output_dir = root / "runs" / "test-predictions"
            run_prediction([
                "--config", str(root / "configs" / "other.yaml"),
                "--weights", str(checkpoint),
                "--output-dir", str(output_dir),
            ])

            result = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
            metadata = json.loads((output_dir / "prediction_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_type"], "AnimalViTPose")
            self.assertEqual(metadata["split"], "val")
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]["predictions"]), 1)
            self.assertTrue(Path(result[0]["output"]).is_file())

            export_dir = root / "runs" / "test-export"
            run_export([
                "--config", str(root / "configs" / "other.yaml"),
                "--weights", str(checkpoint),
                "--output-dir", str(export_dir),
                "--format", "torchscript",
            ])
            torchscript_weights = export_dir / "test-weights.torchscript"
            self.assertTrue(torchscript_weights.is_file())
            exported_prediction_dir = root / "runs" / "exported-model-predictions"
            run_prediction([
                "--config", str(root / "configs" / "other.yaml"),
                "--weights", str(torchscript_weights),
                "--split", "val",
                "--output-dir", str(exported_prediction_dir),
            ])
            exported_results = json.loads(
                (exported_prediction_dir / "predictions.json").read_text(encoding="utf-8")
            )
            exported_metadata = json.loads(
                (exported_prediction_dir / "prediction_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exported_metadata["model_format"], "torchscript")
            self.assertEqual(len(exported_results[0]["predictions"]), 1)

            project.other_config["format"] = "onnx"
            project.other_config["simplify"] = True
            project.save_configs("other")
            onnx_config, onnx_context = load_project_context(root / "configs" / "other.yaml")
            self.assertEqual(onnx_context["other"].get("format"), "onnx")
            onnx_dir = root / "runs" / "test-onnx-export"
            run_export([
                "--config", str(root / "configs" / "other.yaml"),
                "--weights", str(checkpoint),
                "--output-dir", str(onnx_dir),
            ])
            onnx_weights = onnx_dir / "test-weights.onnx"
            self.assertTrue(onnx_weights.is_file())
            onnx_prediction_dir = root / "runs" / "onnx-predictions"
            run_prediction([
                "--config", str(root / "configs" / "other.yaml"),
                "--weights", str(onnx_weights),
                "--split", "val",
                "--output-dir", str(onnx_prediction_dir),
            ])
            onnx_results = json.loads(
                (onnx_prediction_dir / "predictions.json").read_text(encoding="utf-8")
            )
            onnx_metadata = json.loads(
                (onnx_prediction_dir / "prediction_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(onnx_metadata["model_format"], "onnx")
            self.assertEqual(len(onnx_results[0]["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
