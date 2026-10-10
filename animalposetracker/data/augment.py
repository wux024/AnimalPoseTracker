"""Pose-aware image augmentation used by the local training data pipeline."""

import random
from typing import Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _box_xywh_to_xyxy(boxes: np.ndarray, size: int) -> np.ndarray:
    if not boxes.size:
        return np.zeros((0, 4), dtype=np.float32)
    centers = boxes[:, :2] * size
    half_sizes = boxes[:, 2:4] * size * 0.5
    return np.concatenate((centers - half_sizes, centers + half_sizes), axis=1).astype(np.float32)


def _box_xyxy_to_xywh(boxes: np.ndarray, size: int) -> np.ndarray:
    if not boxes.size:
        return np.zeros((0, 4), dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:4]) * 0.5 / size
    dimensions = (boxes[:, 2:4] - boxes[:, :2]) / size
    return np.concatenate((centers, dimensions), axis=1).astype(np.float32)


class PoseAugment:
    """Apply the detection-style HSV, mosaic, affine, and flip transforms to pose labels."""

    def __init__(self, image_size: int, settings: Dict, flip_idx: Optional[Sequence[int]] = None):
        self.image_size = int(image_size)
        self.settings = dict(settings)
        self.flip_idx = tuple(map(int, flip_idx)) if flip_idx is not None else None

    def apply(
        self,
        sample: Dict[str, np.ndarray],
        sample_loader: Callable[[int], Dict[str, np.ndarray]],
        sample_count: int,
        allow_mosaic: bool = True,
        mosaic_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Augment one base sample and, when configured, blend it with a second sample."""
        current = self._apply_geometry(
            sample, sample_loader, sample_count, allow_mosaic, mosaic_indices
        )
        mixup = float(self.settings.get("mixup", 0.0)) if allow_mosaic else 0.0
        if mixup > 0 and np.random.random() < mixup and sample_count > 1:
            second_index = random.randint(0, sample_count - 1)
            second = self._apply_geometry(
                sample_loader(second_index), sample_loader, sample_count, allow_mosaic,
                mosaic_indices,
            )
            blend = float(np.random.beta(32.0, 32.0))
            current["image"] = np.clip(
                current["image"].astype(np.float32) * blend
                + second["image"].astype(np.float32) * (1.0 - blend),
                0,
                255,
            ).astype(np.uint8)
            current["classes"] = np.concatenate((current["classes"], second["classes"]))
            current["boxes"] = np.concatenate((current["boxes"], second["boxes"]))
            current["keypoints"] = np.concatenate((current["keypoints"], second["keypoints"]))
            current["paths"] = [*current["paths"], *second["paths"]]
        image = self._hsv(current["image"])
        image, boxes, keypoints = self._flips(image, current["boxes"], current["keypoints"])
        if float(self.settings.get("bgr", 0.0)) > 0 and np.random.random() < float(self.settings["bgr"]):
            image = image[..., ::-1]
        return {
            "image": np.ascontiguousarray(image),
            "classes": current["classes"].astype(np.int64, copy=False),
            "boxes": _box_xyxy_to_xywh(boxes, self.image_size),
            "keypoints": self._normalize_keypoints(keypoints),
            "paths": current["paths"],
        }

    def _apply_geometry(
        self,
        sample: Dict[str, np.ndarray],
        sample_loader: Callable[[int], Dict[str, np.ndarray]],
        sample_count: int,
        allow_mosaic: bool,
        mosaic_indices: Optional[Sequence[int]],
    ) -> Dict[str, np.ndarray]:
        mosaic_probability = float(self.settings.get("mosaic", 0.0))
        if allow_mosaic and mosaic_probability > 0 and sample_count > 1 and random.random() < mosaic_probability:
            candidates = list(mosaic_indices) if mosaic_indices is not None else list(range(sample_count))
            others = [sample_loader(index) for index in random.choices(candidates, k=3)]
            image, classes, boxes, keypoints, paths = self._mosaic([sample, *others])
        else:
            image = sample["image"].copy()
            classes = sample["classes"].copy()
            boxes = _box_xywh_to_xyxy(sample["boxes"], self.image_size)
            keypoints = sample["keypoints"].copy()
            if keypoints.size:
                keypoints[..., 0] *= self.image_size
                keypoints[..., 1] *= self.image_size
            paths = [sample["path"]]

        image, boxes, keypoints, keep = self._random_perspective(image, boxes, keypoints)
        classes = classes[keep]
        return {
            "image": np.ascontiguousarray(image),
            "classes": classes.astype(np.int64, copy=False),
            "boxes": boxes,
            "keypoints": keypoints,
            "paths": paths,
        }

    def _mosaic(self, samples):
        size = self.image_size
        center_x = int(np.random.uniform(size * 0.5, size * 1.5))
        center_y = int(np.random.uniform(size * 0.5, size * 1.5))
        canvas = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        all_classes, all_boxes, all_keypoints, paths = [], [], [], []
        for index, sample in enumerate(samples):
            height, width = sample["image"].shape[:2]
            if index == 0:
                x1a, y1a, x2a, y2a = max(center_x - width, 0), max(center_y - height, 0), center_x, center_y
                x1b, y1b, x2b, y2b = width - (x2a - x1a), height - (y2a - y1a), width, height
            elif index == 1:
                x1a, y1a, x2a, y2a = center_x, max(center_y - height, 0), min(center_x + width, size * 2), center_y
                x1b, y1b, x2b, y2b = 0, height - (y2a - y1a), x2a - x1a, height
            elif index == 2:
                x1a, y1a, x2a, y2a = max(center_x - width, 0), center_y, center_x, min(center_y + height, size * 2)
                x1b, y1b, x2b, y2b = width - (x2a - x1a), 0, width, y2a - y1a
            else:
                x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + width, size * 2), min(center_y + height, size * 2)
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

            canvas[y1a:y2a, x1a:x2a] = sample["image"][y1b:y2b, x1b:x2b]
            boxes = _box_xywh_to_xyxy(sample["boxes"], size)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + x1a - x1b, x1a, x2a)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + y1a - y1b, y1a, y2a)
            keypoints = sample["keypoints"].copy()
            if keypoints.size:
                keypoints[..., 0] = keypoints[..., 0] * size + x1a - x1b
                keypoints[..., 1] = keypoints[..., 1] * size + y1a - y1b
                outside = (
                    (keypoints[..., 0] < x1a) | (keypoints[..., 0] > x2a)
                    | (keypoints[..., 1] < y1a) | (keypoints[..., 1] > y2a)
                )
                keypoints[..., 0] = keypoints[..., 0].clip(x1a, x2a)
                keypoints[..., 1] = keypoints[..., 1].clip(y1a, y2a)
                if keypoints.shape[-1] == 3:
                    keypoints[..., 2][outside] = 0
            all_classes.append(sample["classes"])
            all_boxes.append(boxes)
            all_keypoints.append(keypoints)
            paths.append(sample["path"])

        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), np.float32)
        keypoints = np.concatenate(all_keypoints, axis=0) if all_keypoints else np.zeros((0, 0, 3), np.float32)
        classes = np.concatenate(all_classes, axis=0) if all_classes else np.zeros((0,), np.int64)
        return canvas, classes, boxes, keypoints, paths

    def _random_perspective(self, image, boxes, keypoints):
        size = self.image_size
        input_height, input_width = image.shape[:2]
        center = np.eye(3, dtype=np.float32)
        center[0, 2], center[1, 2] = -input_width / 2, -input_height / 2
        perspective = np.eye(3, dtype=np.float32)
        perspective[2, 0] = np.random.uniform(-float(self.settings.get("perspective", 0.0)), float(self.settings.get("perspective", 0.0)))
        perspective[2, 1] = np.random.uniform(-float(self.settings.get("perspective", 0.0)), float(self.settings.get("perspective", 0.0)))
        rotation = np.eye(3, dtype=np.float32)
        angle = np.random.uniform(-float(self.settings.get("degrees", 0.0)), float(self.settings.get("degrees", 0.0)))
        scale = np.random.uniform(1.0 - float(self.settings.get("scale", 0.0)), 1.0 + float(self.settings.get("scale", 0.0)))
        rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)
        shear = np.eye(3, dtype=np.float32)
        shear_x = np.random.uniform(-float(self.settings.get("shear", 0.0)), float(self.settings.get("shear", 0.0)))
        shear_y = np.random.uniform(-float(self.settings.get("shear", 0.0)), float(self.settings.get("shear", 0.0)))
        shear[0, 1] = np.tan(np.deg2rad(shear_x))
        shear[1, 0] = np.tan(np.deg2rad(shear_y))
        translate = np.eye(3, dtype=np.float32)
        translate[0, 2] = np.random.uniform(0.5 - float(self.settings.get("translate", 0.0)), 0.5 + float(self.settings.get("translate", 0.0))) * size
        translate[1, 2] = np.random.uniform(0.5 - float(self.settings.get("translate", 0.0)), 0.5 + float(self.settings.get("translate", 0.0))) * size
        matrix = translate @ shear @ rotation @ perspective @ center
        if float(self.settings.get("perspective", 0.0)):
            image = cv2.warpPerspective(image, matrix, dsize=(size, size), borderValue=(114, 114, 114))
        else:
            image = cv2.warpAffine(image, matrix[:2], dsize=(size, size), borderValue=(114, 114, 114))
        if not boxes.size:
            return image, boxes.reshape(0, 4), keypoints, np.zeros((0,), dtype=bool)

        original = boxes.copy()
        corners = np.ones((boxes.shape[0] * 4, 3), dtype=np.float32)
        corners[0::4, :2] = boxes[:, [0, 1]]
        corners[1::4, :2] = boxes[:, [2, 1]]
        corners[2::4, :2] = boxes[:, [2, 3]]
        corners[3::4, :2] = boxes[:, [0, 3]]
        transformed = corners @ matrix.T
        if float(self.settings.get("perspective", 0.0)):
            transformed = transformed[:, :2] / transformed[:, 2:3].clip(min=1e-6)
        else:
            transformed = transformed[:, :2]
        transformed = transformed.reshape(-1, 4, 2)
        candidate_boxes = np.concatenate((transformed.min(axis=1), transformed.max(axis=1)), axis=1)
        if keypoints.size:
            keypoint_xy = keypoints[..., :2]
            homogeneous = np.concatenate((keypoint_xy, np.ones((*keypoint_xy.shape[:-1], 1), np.float32)), axis=-1)
            keypoint_transformed = homogeneous @ matrix.T
            if float(self.settings.get("perspective", 0.0)):
                keypoint_transformed = keypoint_transformed[..., :2] / keypoint_transformed[..., 2:3].clip(min=1e-6)
            else:
                keypoint_transformed = keypoint_transformed[..., :2]
            keypoints[..., :2] = keypoint_transformed
        boxes, keypoints, keep = self._clip_and_filter(
            candidate_boxes, keypoints, original_boxes=original * scale
        )
        return image, boxes, keypoints, keep

    def _clip_and_filter(self, boxes, keypoints, original_boxes=None):
        size = self.image_size
        if not boxes.size:
            return boxes.reshape(0, 4), keypoints, np.zeros((0,), dtype=bool)
        before_clip = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, size)
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        reference_boxes = before_clip if original_boxes is None else original_boxes
        reference_width = reference_boxes[:, 2] - reference_boxes[:, 0]
        reference_height = reference_boxes[:, 3] - reference_boxes[:, 1]
        area_ratio = width * height / (reference_width * reference_height + 1e-16)
        aspect = np.maximum(width / (height + 1e-16), height / (width + 1e-16))
        keep = (width > 2) & (height > 2) & (area_ratio > 0.1) & (aspect < 100)
        if keypoints.size:
            outside = (
                (keypoints[..., 0] < 0) | (keypoints[..., 0] > size)
                | (keypoints[..., 1] < 0) | (keypoints[..., 1] > size)
            )
            keypoints[..., 0] = keypoints[..., 0].clip(0, size)
            keypoints[..., 1] = keypoints[..., 1].clip(0, size)
            if keypoints.shape[-1] == 3:
                keypoints[..., 2][outside] = 0
        return boxes[keep], keypoints[keep] if keypoints.size else keypoints, keep

    def _hsv(self, image):
        hue, saturation, value = (
            float(self.settings.get(name, 0.0)) for name in ("hsv_h", "hsv_s", "hsv_v")
        )
        if not (hue or saturation or value):
            return image
        random_gains = np.random.uniform(-1.0, 1.0, 3) * np.asarray(
            [hue, saturation, value], dtype=np.float64
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        x = np.arange(256, dtype=random_gains.dtype)
        lut_hue = ((x + random_gains[0] * 180) % 180).astype(np.uint8)
        lut_sat = np.clip(x * (random_gains[1] + 1.0), 0, 255).astype(np.uint8)
        lut_sat[0] = 0
        lut_val = np.clip(x * (random_gains[2] + 1.0), 0, 255).astype(np.uint8)
        hsv = cv2.merge((
            cv2.LUT(hsv[:, :, 0], lut_hue),
            cv2.LUT(hsv[:, :, 1], lut_sat),
            cv2.LUT(hsv[:, :, 2], lut_val),
        ))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _flips(self, image, boxes, keypoints):
        size = self.image_size
        if np.random.random() < float(self.settings.get("fliplr", 0.0)):
            image = np.ascontiguousarray(image[:, ::-1])
            if boxes.size:
                x1 = boxes[:, 0].copy()
                boxes[:, 0], boxes[:, 2] = size - boxes[:, 2], size - x1
            if keypoints.size:
                keypoints[..., 0] = size - keypoints[..., 0]
                if self.flip_idx is not None:
                    keypoints = keypoints[:, self.flip_idx, :]
        if np.random.random() < float(self.settings.get("flipud", 0.0)):
            image = np.ascontiguousarray(image[::-1])
            if boxes.size:
                y1 = boxes[:, 1].copy()
                boxes[:, 1], boxes[:, 3] = size - boxes[:, 3], size - y1
            if keypoints.size:
                keypoints[..., 1] = size - keypoints[..., 1]
        return image, boxes, keypoints

    def _normalize_keypoints(self, keypoints):
        if not keypoints.size:
            return keypoints.reshape(0, keypoints.shape[-2] if keypoints.ndim == 3 else 0, keypoints.shape[-1] if keypoints.ndim == 3 else 3).astype(np.float32)
        result = keypoints.astype(np.float32, copy=True)
        result[..., 0] /= self.image_size
        result[..., 1] /= self.image_size
        return result


__all__ = ["PoseAugment"]
