"""Dataset-oriented prediction worker for AnimalRTPose and AnimalViTPose."""

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from animalposetracker.project.model_context import (
    configure_project_model_spec,
    load_project_context,
    load_project_model,
    read_yaml_mapping,
    unique_output_directory,
)
from animalposetracker.artifacts import detect_artifact_format, read_artifact_metadata
from animalposetracker.preprocessing.letterbox import letterbox_image
from animalposetracker.prediction.backends import (
    ModelArtifactBackend,
)


VIDEO_SUFFIXES = {
    ".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".wmv", ".webm",
}
IMAGE_SUFFIXES = {
    ".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo", ".pfm", ".png",
    ".tif", ".tiff", ".webp",
}


def _output_stem(path: Path, peers: List[Path]) -> str:
    """Keep familiar output names while disambiguating equal stems from different folders."""
    path = Path(path).resolve()
    collisions = sum(Path(peer).stem.casefold() == path.stem.casefold() for peer in peers)
    if collisions <= 1:
        return path.stem
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def _letterbox(image: np.ndarray, size: int):
    canvas, scale, left, top = letterbox_image(image, size, size)
    return canvas, scale, float(left), float(top)


def _as_numpy(value):
    detach = getattr(value, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(value)


def _resolve_input_path(value: str, project_dir: Path, dataset: Dict[str, Any]) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    data_root = Path(str(dataset.get("path") or "")).expanduser()
    if not data_root.is_absolute():
        data_root = project_dir / data_root
    candidate = data_root / path
    if candidate.exists():
        return candidate.resolve()
    return (project_dir / path).resolve()


def _source_paths(source: Any, project_dir: Path, dataset: Dict[str, Any]) -> Tuple[List[Path], List[Path]]:
    values = source if isinstance(source, (list, tuple)) else [source]
    images: List[Path] = []
    videos: List[Path] = []
    for value in values:
        raw = str(value).strip()
        path = _resolve_input_path(raw, project_dir, dataset)
        candidates = []
        if any(token in raw for token in ("*", "?", "[")):
            pattern = raw if Path(raw).is_absolute() else str(project_dir / raw)
            candidates = [Path(item).resolve() for item in sorted(glob.glob(pattern, recursive=True))]
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                item = line.strip()
                if item and not item.startswith("#"):
                    candidates.append(_resolve_input_path(item, path.parent, dataset))
        elif path.exists():
            candidates = [path]
        if not candidates:
            raise FileNotFoundError(f"Prediction source does not exist or is unsupported: {raw}")
        for item in candidates:
            if item.is_dir():
                children = sorted(item.rglob("*"))
            else:
                children = [item]
            for child in children:
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(child.resolve())
                elif child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES:
                    videos.append(child.resolve())
    return images, videos


def _resolve_dataset_split(dataset, requested: str):
    """Choose a configured dataset split, preferring test and falling back to val."""
    from animalposetracker.data.pose import PoseTextDataset

    if requested != "auto":
        resolved = PoseTextDataset(dataset, split=requested, image_size=640, cache=False)
        if not resolved.image_paths:
            raise ValueError(f"The {requested} split contains no readable images")
        return resolved
    errors = []
    for split in ("test", "val"):
        if not read_yaml_mapping(dataset).get(split):
            continue
        try:
            resolved = PoseTextDataset(dataset, split=split, image_size=640, cache=False)
            if not resolved.image_paths:
                raise ValueError(f"The {split} split contains no readable images")
            return resolved
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{split}: {exc}")
    message = "; ".join(errors) if errors else "dataset.yaml defines neither test nor val"
    raise ValueError(f"Could not resolve a default prediction split: {message}")


def _expand_dataset_images(source, root: Path, split: str) -> List[Path]:
    """Expand local image sources from a dataset YAML without requiring label files."""
    values = source if isinstance(source, (list, tuple)) else [source]
    images = []
    for value in values:
        if value is None:
            continue
        raw = str(value).strip()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        if any(token in raw for token in ("*", "?", "[")):
            candidates = [Path(item).resolve() for item in glob.glob(str(path), recursive=True)]
        elif path.is_dir():
            candidates = [path.resolve()]
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates = [path.resolve()]
        elif path.is_file() and path.suffix.lower() in {".txt", ".list"}:
            candidates = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                candidate = Path(item).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                    if not candidate.exists():
                        candidate = root / item
                candidates.append(candidate.resolve())
        else:
            continue
        for candidate in candidates:
            paths = candidate.rglob("*") if candidate.is_dir() else [candidate]
            images.extend(
                item.resolve()
                for item in paths
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
    return sorted(set(images))


def _resolve_image_split(data_path: Path, requested: str):
    """Select a dataset image split independently of its annotations."""
    dataset = read_yaml_mapping(data_path)
    root = Path(str(dataset.get("path") or Path(data_path).parent)).expanduser()
    if not root.is_absolute():
        root = Path(data_path).parent / root
    root = root.resolve()
    if requested != "auto":
        splits = [requested]
    else:
        splits = [split for split in ("test", "val") if dataset.get(split)]
    errors = []
    for split in splits:
        image_source = dataset.get(f"{split}_images")
        if image_source is None and isinstance(dataset.get("images"), dict):
            image_source = dataset["images"].get(split)
        source = dataset.get(split)
        source_is_json = isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() == ".json"
        if image_source is None and not source_is_json:
            image_source = source
        image_paths = _expand_dataset_images(image_source, root, split)
        if not image_paths and source_is_json:
            inferred_roots = (
                root / "images" / f"{split}2017",
                root / f"{split}2017",
                root / "images" / split,
                root / split,
                root / "images",
            )
            for candidate in inferred_roots:
                if candidate.is_dir():
                    image_paths = _expand_dataset_images(candidate, root, split)
                    if image_paths:
                        break
        if image_paths:
            return SimpleNamespace(split=split, image_paths=image_paths)
        errors.append(f"{split}: no supported images found")
    message = "; ".join(errors) if errors else "dataset.yaml defines neither test nor val"
    raise ValueError(f"Could not resolve a dataset image split: {message}")


def _make_topdown_dataset(config, context, split):
    from animalposetracker.data.topdown import TopDownPoseDataset

    width, height = context["topdown_input_size"]
    return TopDownPoseDataset(
        config.data,
        split=split,
        input_size=(width, height),
        sigma=float(context["other"].get("label_sigma", 6.0)),
        split_ratio=context["simcc_split_ratio"],
        cache=context["other"].get("cache", False),
        preprocessing_config=context["preprocessing"],
    )


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        from PIL import Image

        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise RuntimeError(f"Could not decode prediction image {path}: {exc}") from exc


def _render_pose(
    image: np.ndarray,
    detections,
    class_names: List[str],
    skeleton: List[List[int]],
    show_boxes: bool,
    show_labels: bool,
    show_keypoints: bool,
    show_skeletons: bool,
    confidence_threshold: float,
    point_radius: int,
    line_width: int,
):
    rendered = image.copy()
    records = []
    colors = [
        (0, 200, 255), (255, 120, 0), (80, 220, 80), (220, 80, 220),
        (255, 220, 0), (80, 200, 220), (200, 120, 255), (80, 160, 255),
        (180, 220, 80), (220, 180, 80), (120, 255, 180), (255, 140, 180),
    ]
    boxes, scores, classes, keypoints = detections
    for box, score, class_id, points in zip(boxes, scores, classes, keypoints):
        box_value = _as_numpy(box).reshape(-1)
        score_value = float(_as_numpy(score).item())
        class_value = int(class_id)
        points_value = _as_numpy(points).astype(np.float32, copy=False)
        x1, y1, x2, y2 = [int(round(float(value))) for value in box_value.tolist()]
        if show_boxes:
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 220, 0), line_width)
        if show_labels:
            class_name = class_names[class_value] if class_value < len(class_names) else str(class_value)
            cv2.putText(
                rendered,
                f"{class_name} {score_value:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 0),
                max(line_width, 1),
                cv2.LINE_AA,
            )
        point_visible = (
            points_value[:, 2] >= confidence_threshold
            if points_value.shape[1] > 2
            else np.ones((points_value.shape[0],), dtype=bool)
        )
        if show_skeletons:
            for edge in skeleton:
                if len(edge) != 2:
                    continue
                left, right = map(int, edge)
                if (
                    0 <= left < len(points_value)
                    and 0 <= right < len(points_value)
                    and point_visible[left]
                    and point_visible[right]
                ):
                    p1 = tuple(np.rint(points_value[left, :2]).astype(int))
                    p2 = tuple(np.rint(points_value[right, :2]).astype(int))
                    cv2.line(rendered, p1, p2, (0, 220, 255), line_width, cv2.LINE_AA)
        if show_keypoints:
            for point_index, point in enumerate(points_value):
                if not point_visible[point_index]:
                    continue
                center = tuple(np.rint(point[:2]).astype(int))
                cv2.circle(
                    rendered,
                    center,
                    point_radius,
                    colors[point_index % len(colors)],
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        records.append({
            "bbox_xyxy": [float(v) for v in box_value.tolist()],
            "score": score_value,
            "class_id": class_value,
            "class_name": class_names[class_value] if class_value < len(class_names) else str(class_value),
            "keypoints": points_value.tolist(),
        })
    return rendered, records


def _predict_frame(image, model, config, context, settings, external_backend=False):
    import torch

    from animalposetracker.postprocessing.nms import pose_non_max_suppression

    canvas, scale, pad_x, pad_y = _letterbox(image, config.image_size)
    tensor = torch.from_numpy(
        np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    ).to(device=context["device"], dtype=torch.float32).div_(255.0).unsqueeze(0)
    if external_backend:
        raw_outputs = model(tensor.detach().cpu().numpy())
        if not raw_outputs:
            raise RuntimeError("The exported pose model returned no output tensors")
        prediction = torch.as_tensor(raw_outputs[0], device=context["device"])
    else:
        with torch.inference_mode():
            output = model(tensor)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError("AnimalRTPose inference expects (decoded predictions, raw predictions)")
        prediction = output[0]
    detections = pose_non_max_suppression(
        prediction.float(),
        num_classes=int(context["model_nc"]),
        num_keypoints=int(context["kpt_shape"][0]),
        keypoint_dimensions=int(context["kpt_shape"][1]),
        confidence_threshold=settings["confidence"],
        iou_threshold=settings["iou"],
        max_detections=settings["max_detections"],
        agnostic=settings["agnostic_nms"],
    )[0]

    boxes, scores, classes, keypoints = detections
    original_height, original_width = image.shape[:2]
    if boxes.numel():
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clamp(0, original_width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clamp(0, original_height)
        keypoints[..., 0] = ((keypoints[..., 0] - pad_x) / scale).clamp(0, original_width)
        keypoints[..., 1] = ((keypoints[..., 1] - pad_y) / scale).clamp(0, original_height)
    detections = (boxes, scores, classes, keypoints)
    rendered, records = _render_pose(
        image,
        detections,
        context["class_names"],
        context["skeleton"],
        show_boxes=settings["show_boxes"],
        show_labels=settings["show_labels"],
        show_keypoints=settings["show_keypoints"],
        show_skeletons=settings["show_skeletons"],
        confidence_threshold=settings["keypoint_confidence"],
        point_radius=settings["point_radius"],
        line_width=settings["line_width"],
    )
    return rendered, records


def _letterbox_shape(image: np.ndarray, width: int, height: int):
    return letterbox_image(image, width, height)


class InstanceBoxProvider:
    """Interface for supplying animal instance boxes to the top-down pose model."""

    def boxes_for_image(self, image: np.ndarray, image_path: Path) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DatasetAnnotationBoxProvider(InstanceBoxProvider):
    """Provide ground-truth boxes from the selected dataset split."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.by_path: Dict[str, List[Dict[str, Any]]] = {}
        category_mapping = dataset.annotations._coco_category_to_class or {}
        for entry in dataset._entries:
            category_id = int(entry["category_id"])
            class_id = category_mapping.get(category_id, category_id - 1)
            self.by_path.setdefault(str(Path(entry["image_path"]).resolve()), []).append({
                "bbox_xyxy": np.asarray(entry["bbox"], dtype=np.float32),
                "class_id": int(class_id),
                "score": 1.0,
            })

    def boxes_for_image(self, image: np.ndarray, image_path: Path) -> List[Dict[str, Any]]:
        del image
        return list(self.by_path.get(str(Path(image_path).resolve()), []))


class DetectorBoxProvider(InstanceBoxProvider):
    """Adapter seam for a future detector that returns source-image xyxy boxes."""

    def __init__(self, detector) -> None:
        self.detector = detector

    def boxes_for_image(self, image: np.ndarray, image_path: Path) -> List[Dict[str, Any]]:
        del image_path
        if hasattr(self.detector, "predict_boxes"):
            return list(self.detector.predict_boxes(image))
        if callable(self.detector):
            return list(self.detector(image))
        raise TypeError("A detector provider must be callable or implement predict_boxes(image)")


def _predict_topdown_image(image, image_path, boxes, model, context, settings, external_backend=False):
    """Predict all supplied instance boxes using AnimalViTPose's evaluation geometry."""
    import torch

    from animalposetracker.preprocessing.topdown import _fix_aspect_ratio, _topdown_warp_matrix

    original = image
    rendered = image.copy()
    records = []
    width, height = context["topdown_input_size"]
    padding = float(context["preprocessing"].get("bbox_padding", 1.25))
    mean = np.asarray(
        context["preprocessing"].get("pixel_mean", [123.675, 116.28, 103.53]),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    std = np.asarray(
        context["preprocessing"].get("pixel_std", [58.395, 57.12, 57.375]),
        dtype=np.float32,
    ).reshape(1, 1, 3)
    ratio = float(context["simcc_split_ratio"])

    for box_record in boxes:
        box = np.asarray(box_record["bbox_xyxy"], dtype=np.float32).reshape(4)
        center = (box[:2] + box[2:]) * 0.5
        scale = (box[2:] - box[:2]) * padding
        scale = _fix_aspect_ratio(scale, (width, height))
        warp = _topdown_warp_matrix(center, scale, 0.0, (width, height))
        inverse = cv2.invertAffineTransform(warp).astype(np.float32)
        crop = cv2.warpAffine(original, warp, (width, height), flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        crop = (crop - mean) / std
        batch = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32)
        if external_backend:
            raw_outputs = model(batch)
            if len(raw_outputs) < 2:
                raise RuntimeError("AnimalViTPose inference expects SimCC x and y output tensors")
            pred_x = torch.as_tensor(raw_outputs[0])
            pred_y = torch.as_tensor(raw_outputs[1])
        else:
            tensor = torch.from_numpy(batch).to(context["device"])
            with torch.inference_mode():
                pred_x, pred_y = model(tensor)
            pred_x = pred_x.detach().cpu()
            pred_y = pred_y.detach().cpu()

        max_x, coords_x = pred_x[0].max(dim=-1)
        max_y, coords_y = pred_y[0].max(dim=-1)
        keypoint_scores = torch.minimum(max_x, max_y).numpy()
        coordinates = torch.stack((coords_x, coords_y), dim=-1).float().numpy() / ratio
        invalid = keypoint_scores <= 0
        coordinates[invalid] = -1.0 / ratio
        homogeneous = np.concatenate(
            [coordinates, np.ones((len(coordinates), 1), dtype=np.float32)], axis=-1
        )
        source_coordinates = np.einsum("ij,kj->ki", inverse, homogeneous)
        keypoints = np.concatenate(
            [source_coordinates, keypoint_scores[:, None]], axis=1
        ).astype(np.float32)
        detection = (
            box.reshape(1, 4),
            np.asarray([float(keypoint_scores.mean())], dtype=np.float32),
            np.asarray([int(box_record.get("class_id", 0))], dtype=np.int64),
            keypoints[None],
        )
        rendered, instance_records = _render_pose(
            rendered,
            detection,
            context["class_names"],
            context["skeleton"],
            show_boxes=settings["show_boxes"],
            show_labels=settings["show_labels"],
            show_keypoints=settings["show_keypoints"],
            show_skeletons=settings["show_skeletons"],
            confidence_threshold=settings["keypoint_confidence"],
            point_radius=settings["point_radius"],
            line_width=settings["line_width"],
        )
        records.extend(instance_records)
    return rendered, records


def _write_video(
    source_path,
    output_path,
    frame_predictor,
    all_records,
    tracker=None,
    class_names=None,
):
    from animalposetracker.postprocessing.workflows import track_frame_predictions

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source {source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video {output_path}")
    frame_index = 0
    processed_records = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rendered, records = frame_predictor(frame)
            if tracker is not None:
                tracked_records = track_frame_predictions(
                    records,
                    tracker,
                    frame_index,
                    frame=frame,
                    class_names=class_names,
                )
                for detection in tracked_records:
                    box = detection.get("bbox_xyxy")
                    if box is not None:
                        x1, y1 = int(round(box[0])), int(round(box[1]))
                        cv2.putText(
                            rendered,
                            f"ID {detection['track_id']}",
                            (x1, max(18, y1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                processed_records.append({
                    "source": str(source_path),
                    "frame": frame_index,
                    "predictions": tracked_records,
                })
            writer.write(rendered)
            all_records.append({
                "source": str(source_path),
                "frame": frame_index,
                "predictions": records,
            })
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return frame_index, processed_records


def _make_argument_parser():
    from animalposetracker.tracking.config import ALGORITHMS

    parser = argparse.ArgumentParser(
        prog="animalpose-cli predict",
        description="Predict AnimalRTPose or AnimalViTPose on a project dataset split.",
    )
    parser.add_argument("--config", default="configs/other.yaml")
    parser.add_argument("--weights", required=True, help="Pose checkpoint, exported artifact, or Triton model URL")
    parser.add_argument(
        "--source",
        nargs="+",
        help="Optional local image/video paths, directories, globs, or text manifests",
    )
    parser.add_argument(
        "--split",
        choices=("auto", "train", "val", "test"),
        default="auto",
        help="Dataset split used by default; auto prefers test and falls back to val",
    )
    parser.add_argument("--output-dir", help="Override the prediction output directory")
    parser.add_argument(
        "--tracker",
        choices=ALGORITHMS,
        help="Optionally track video detections with this AnimalPoseTracker algorithm",
    )
    parser.add_argument(
        "--pose-filter",
        choices=("none", "median", "savgol"),
        help="Optionally smooth tracked video keypoints after prediction",
    )
    parser.add_argument("--filter-window", type=int, help="Temporal filter window length")
    parser.add_argument("--filter-polyorder", type=int, help="Savitzky-Golay polynomial order")
    return parser


def _print_predict_banner(
    *,
    weights_path,
    artifact_format: str,
    context: Dict[str, Any],
    head_name: str,
    image_count: int,
    video_count: int,
    tracker_name,
    filter_method: str,
    output_dir: Path,
) -> None:
    if not sys.stdout.isatty():
        return
    from animalposetracker.training.console import environment_line, format_summary_block

    project = context.get("project") or {}
    model_type = str(project.get("model_type") or "model")
    model_scale = str(project.get("model_scale") or "")
    source_parts = []
    if image_count:
        source_parts.append(f"{image_count} images")
    if video_count:
        source_parts.append(f"{video_count} videos")
    split = context.get("prediction_split")
    if split:
        source_parts.append(f"split={split}")
    tracking = str(tracker_name) if tracker_name else "off"
    if tracker_name and filter_method != "none":
        tracking += f" + {filter_method} filter"
    rows = [
        ("Model", f"{model_type}-{model_scale}".rstrip("-") + f" ({artifact_format})"),
        ("Weights", str(weights_path)),
        ("Head", str(head_name)),
        ("Source", " · ".join(source_parts) if source_parts else "dataset split"),
        ("Tracking", tracking),
        ("Output", str(output_dir)),
    ]
    print()
    print(environment_line())
    print()
    print(format_summary_block(rows))
    print()


def _print_progress(text: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\r" + text.ljust(110))
        sys.stdout.flush()


def _end_progress() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()


def run(argv=None, box_provider: InstanceBoxProvider = None) -> int:
    import torch

    args = _make_argument_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    remote_model = str(args.weights).startswith(("http://", "grpc://"))
    weights_path = (
        str(args.weights) if remote_model else Path(args.weights).expanduser().resolve()
    )
    artifact_metadata = {} if remote_model else read_artifact_metadata(weights_path)
    artifact_format = detect_artifact_format(weights_path, artifact_metadata)

    if artifact_format == "pt":
        model, config, context = load_project_model(config_path, weights_path)
        context["device"] = model.device if hasattr(model, "device") else next(model.parameters()).device
        context["model_nc"] = int(model.nc)
        context["kpt_shape"] = tuple(model.kpt_shape)
        head = model.model[-1]
        head_name = context["head_name"]
        if head_name == "SimCCHead":
            context["topdown_input_size"] = tuple(map(int, head.input_size))
            context["simcc_split_ratio"] = float(head.simcc_split_ratio)
    else:
        config, context = load_project_context(config_path)
        model_spec, head_name, scale = configure_project_model_spec(config, context)
        context.update({
            "model_spec": model_spec,
            "head_name": head_name,
            "scale": scale,
            "model_nc": int(model_spec.get("nc", 1)),
            "kpt_shape": tuple(model_spec.get("kpt_shape") or context["dataset"].get("kpt_shape")),
            "device": __import__("animalposetracker.project.model_context", fromlist=["resolve_device"])
            .resolve_device(torch, config.device),
        })
        model = ModelArtifactBackend(
            weights_path,
            artifact_format,
            context["device"],
            metadata=artifact_metadata,
            use_opencv_dnn=bool(context["other"].get("dnn", False)),
        )
        if head_name == "SimCCHead":
            head_args = model_spec["head"][-1][3]
            context["topdown_input_size"] = tuple(map(int, head_args[1]))
            context["simcc_split_ratio"] = float(head_args[3])

    if head_name not in {"YOLOPoseHead", "SimCCHead"}:
        raise NotImplementedError(f"Project prediction does not support model head {head_name!r}")
    dataset_names = context["dataset"].get("names") or {}
    dataset_names = list(dataset_names.values()) if isinstance(dataset_names, dict) else list(dataset_names)
    context["class_names"] = context["project"].get("classes_name") or dataset_names
    context["skeleton"] = context["project"].get("skeleton") or context["dataset"].get("skeleton") or []

    metadata_input_size = artifact_metadata.get("input_size")
    if head_name == "YOLOPoseHead" and metadata_input_size:
        config.image_size = int(metadata_input_size[0])
    elif head_name == "SimCCHead" and metadata_input_size:
        context["topdown_input_size"] = tuple(map(int, metadata_input_size[:2]))
    if head_name == "SimCCHead":
        context["preprocessing"] = dict(context["model_spec"].get("preprocessing") or {})

    project_dir = context["project_dir"]
    source_value = args.source or context["other"].get("source")
    dataset_predict = source_value is None
    split_request = args.split
    if split_request == "auto" and context["other"].get("predict_split"):
        split_request = str(context["other"]["predict_split"])
    data_dataset = None
    topdown_dataset = None
    if head_name == "SimCCHead":
        if config.data is None:
            raise ValueError("Project prediction requires a dataset.yaml path")
        data_dataset = _resolve_dataset_split(config.data, split_request)
        context["prediction_split"] = data_dataset.split
        if head_name == "SimCCHead":
            if box_provider is not None and not hasattr(box_provider, "boxes_for_image"):
                box_provider = DetectorBoxProvider(box_provider)
            if source_value is None or box_provider is None:
                try:
                    topdown_dataset = _make_topdown_dataset(config, context, data_dataset.split)
                except (FileNotFoundError, ValueError):
                    if split_request != "auto" or data_dataset.split != "test":
                        raise
                    data_dataset = _resolve_dataset_split(config.data, "val")
                    topdown_dataset = _make_topdown_dataset(config, context, data_dataset.split)
                    context["prediction_split"] = data_dataset.split
                context["topdown_dataset"] = topdown_dataset
            if source_value is not None:
                selected_images, selected_videos = _source_paths(
                    source_value, project_dir, context["dataset"]
                )
                if selected_videos and box_provider is None:
                    raise ValueError(
                        "AnimalViTPose dataset prediction needs annotated instance boxes; "
                        "external video prediction requires a detector box provider."
                    )
                if box_provider is not None:
                    image_paths, videos = selected_images, selected_videos
                else:
                    selected = {str(path.resolve()) for path in selected_images}
                    topdown_dataset._entries = [
                        entry for entry in topdown_dataset._entries
                        if str(Path(entry["image_path"]).resolve()) in selected
                    ]
                    if not topdown_dataset._entries:
                        raise ValueError(
                            "The supplied source has no annotated instances in the selected dataset split; "
                            "pass a detector box provider for external images."
                        )
                    image_paths = sorted({
                        Path(entry["image_path"]).resolve() for entry in topdown_dataset._entries
                    })
                    videos = []
            else:
                if box_provider is None:
                    box_provider = DatasetAnnotationBoxProvider(topdown_dataset)
                image_paths = sorted({
                    Path(entry["image_path"]).resolve() for entry in topdown_dataset._entries
                })
                videos = []
            if box_provider is None:
                box_provider = DatasetAnnotationBoxProvider(topdown_dataset)
            context["box_provider"] = box_provider
        else:
            image_paths, videos = _source_paths(source_value, project_dir, context["dataset"])
    elif dataset_predict:
        if config.data is None:
            raise ValueError("Project prediction requires a dataset.yaml path")
        data_dataset = _resolve_image_split(config.data, split_request)
        context["prediction_split"] = data_dataset.split
        image_paths = [Path(path).resolve() for path in data_dataset.image_paths]
        videos = []
    else:
        image_paths, videos = _source_paths(source_value, project_dir, context["dataset"])
        if not image_paths and not videos:
            raise ValueError("The prediction source contains no supported images or videos")
    if head_name == "YOLOPoseHead" and not image_paths and not videos:
        raise ValueError("The selected dataset split contains no supported images or videos")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        config.output_dir = (
            output_dir if output_dir.is_absolute()
            else context["project_dir"] / output_dir
        ).resolve()
    else:
        # Prediction runs get their own namespace instead of inheriting the
        # training run name (project/name = runs/train) from other.yaml.
        config.output_dir = context["project_dir"] / "runs" / "predict"
    config.output_dir = unique_output_directory(
        config.output_dir,
        exist_ok=bool(context["other"].get("exist_ok", False)),
    )

    settings_values = context["other"]
    settings = {
        "confidence": 0.25 if settings_values.get("conf") is None else float(settings_values["conf"]),
        "keypoint_confidence": float(settings_values.get("keypoint_score_threshold", 0.25)),
        "iou": float(settings_values.get("iou", 0.7)),
        "max_detections": int(settings_values.get("max_det", 300)),
        "agnostic_nms": bool(settings_values.get("agnostic_nms", False)),
        "show_boxes": bool(settings_values.get("show_boxes", True)),
        "show_labels": bool(settings_values.get("show_labels", True)),
        "show_keypoints": bool(settings_values.get("show_keypoints", True)),
        "show_skeletons": bool(settings_values.get("kpt_line", False)),
        "point_radius": int(settings_values.get("kpt_radius") or 4),
        "line_width": int(settings_values.get("line_width") or 2),
    }
    tracking_value = settings_values.get("tracking") or {}
    if not isinstance(tracking_value, dict):
        raise TypeError("other.yaml tracking settings must be a mapping")
    tracking_settings = dict(tracking_value)
    tracking_enabled = bool(tracking_settings.pop("enabled", False))
    configured_tracker = tracking_settings.pop("algorithm", None)
    tracker_name = args.tracker or (
        configured_tracker if tracking_enabled else None
    )
    filter_value = settings_values.get("pose_filter") or {}
    if isinstance(filter_value, str):
        filter_value = {"method": filter_value}
    if not isinstance(filter_value, dict):
        raise TypeError("other.yaml pose_filter settings must be a mapping or filter name")
    pose_filter_settings = dict(filter_value)
    filter_method = str(
        args.pose_filter or pose_filter_settings.get("method", "none")
    ).lower()
    if filter_method not in {"none", "median", "savgol"}:
        raise ValueError("pose_filter.method must be one of: none, median, savgol")
    if filter_method != "none" and tracker_name is None:
        tracker_name = "bytetrack"
    filter_window = int(
        args.filter_window if args.filter_window is not None
        else pose_filter_settings.get("window_length", 5)
    )
    filter_polyorder = int(
        args.filter_polyorder if args.filter_polyorder is not None
        else pose_filter_settings.get("polyorder", 2)
    )
    filter_confidence = pose_filter_settings.get(
        "confidence_threshold", settings["keypoint_confidence"]
    )
    filter_max_gap = int(pose_filter_settings.get("max_gap", 0))
    if tracker_name and not videos:
        raise ValueError("Video tracking requires at least one video source")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_predict_banner(
        weights_path=weights_path,
        artifact_format=artifact_format,
        context=context,
        head_name=head_name,
        image_count=len(image_paths),
        video_count=len(videos),
        tracker_name=tracker_name,
        filter_method=filter_method,
        output_dir=output_dir,
    )
    results = []
    tracked_results = []
    filtered_results = []

    external_backend = artifact_format != "pt"
    image_total = len(image_paths)
    if head_name == "YOLOPoseHead":
        for image_index, image_path in enumerate(image_paths, 1):
            _print_progress(f"predict {image_index}/{image_total} {image_path.name}")
            image = _read_image(image_path)
            rendered, records = _predict_frame(
                image, model, config, context, settings, external_backend=external_backend
            )
            output_path = output_dir / "images" / f"{_output_stem(image_path, image_paths)}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), rendered):
                raise RuntimeError(f"Could not save prediction image {output_path}")
            results.append({"source": str(image_path), "output": str(output_path), "predictions": records})

    else:
        provider = context["box_provider"]
        for image_index, image_path in enumerate(image_paths, 1):
            _print_progress(f"predict {image_index}/{image_total} {image_path.name}")
            image = _read_image(image_path)
            boxes = provider.boxes_for_image(image, image_path)
            rendered, records = _predict_topdown_image(
                image,
                image_path,
                boxes,
                model,
                context,
                settings,
                external_backend=external_backend,
            )
            output_path = output_dir / "images" / f"{_output_stem(image_path, image_paths)}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), rendered):
                raise RuntimeError(f"Could not save prediction image {output_path}")
            results.append({"source": str(image_path), "output": str(output_path), "predictions": records})

    for video_path in videos:
        _print_progress(f"video {video_path.name}")
        output_path = output_dir / "videos" / f"{_output_stem(video_path, videos)}.mp4"
        if head_name == "YOLOPoseHead":
            frame_predictor = lambda frame: _predict_frame(
                frame, model, config, context, settings, external_backend=external_backend
            )
        else:
            if box_provider is None:
                raise ValueError("AnimalViTPose video prediction requires an instance box provider")
            frame_predictor = lambda frame, path=video_path: _predict_topdown_image(
                frame,
                path,
                box_provider.boxes_for_image(frame, path),
                model,
                context,
                settings,
                external_backend=external_backend,
            )
        tracker = None
        processed_records = []
        if tracker_name:
            from animalposetracker.tracking import create_tracker

            tracker = create_tracker({**tracking_settings, "algorithm": tracker_name})
        frame_count, processed_records = _write_video(
            video_path,
            output_path,
            frame_predictor,
            results,
            tracker=tracker,
            class_names=context["class_names"],
        )
        video_summary = {
            "source": str(video_path),
            "output": str(output_path),
            "frames": frame_count,
        }
        if tracker is not None:
            tracked_results.extend(processed_records)
            tracked_results.append(video_summary)
            if filter_method != "none":
                from animalposetracker.postprocessing.workflows import filter_tracked_video_records

                filtered_records = filter_tracked_video_records(
                    processed_records,
                    method=filter_method,
                    window_length=filter_window,
                    polyorder=filter_polyorder,
                    confidence_threshold=filter_confidence,
                    max_gap=filter_max_gap,
                )
                filtered_results.extend(filtered_records)
                filtered_results.append(video_summary)
        results.append(video_summary)

    results_path = output_dir / "predictions.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if tracked_results:
        (output_dir / "tracked_predictions.json").write_text(
            json.dumps(tracked_results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if filtered_results:
        (output_dir / "filtered_predictions.json").write_text(
            json.dumps(filtered_results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    metadata_path = output_dir / "prediction_metadata.json"
    metadata_path.write_text(
        json.dumps({
            "model": str(weights_path),
            "model_format": artifact_format,
            "model_type": context["project"].get("model_type"),
            "split": context.get("prediction_split"),
            "tracker": tracker_name,
            "pose_filter": (
                None if filter_method == "none" else {
                    "method": filter_method,
                    "window_length": filter_window,
                    "polyorder": filter_polyorder,
                    "confidence_threshold": filter_confidence,
                    "max_gap": filter_max_gap,
                }
            ),
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _end_progress()
    print(f"Predictions saved to {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
