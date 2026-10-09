"""COCO Keypoints JSON indexing for the local pose data pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _category_map(categories, class_names: Optional[Sequence[str]]):
    ordered_categories = sorted(categories, key=lambda item: int(item.get("id", -1)))
    if not class_names:
        names = [str(category.get("name", category["id"])) for category in ordered_categories]
        return names, {int(category["id"]): index for index, category in enumerate(ordered_categories)}

    names = [str(name) for name in class_names]
    names_to_index = {name.strip().casefold(): index for index, name in enumerate(names)}
    mapping = {
        int(category["id"]): names_to_index[str(category.get("name", "")).strip().casefold()]
        for category in ordered_categories
        if str(category.get("name", "")).strip().casefold() in names_to_index
    }
    if not mapping and len(ordered_categories) == len(names):
        mapping = {
            int(category["id"]): index for index, category in enumerate(ordered_categories)
        }
    if not mapping:
        raise ValueError(
            "COCO categories do not match dataset names; provide matching names or one name per category"
        )
    return names, mapping


def _coco_keypoint_shape(categories, annotations, requested_shape):
    inferred_count = 0
    for category in categories:
        if category.get("keypoints"):
            inferred_count = len(category["keypoints"])
            break
    if not inferred_count:
        for annotation in annotations:
            values = annotation.get("keypoints")
            if values:
                if len(values) % 3:
                    raise ValueError("COCO keypoints must contain x, y, visibility triplets")
                inferred_count = len(values) // 3
                break
    if requested_shape is None:
        if inferred_count < 1:
            raise ValueError("COCO Keypoints JSON needs keypoints metadata or dataset kpt_shape")
        return inferred_count, 3
    if not isinstance(requested_shape, (list, tuple)) or len(requested_shape) != 2:
        raise ValueError("Dataset kpt_shape must be [keypoint_count, dimensions]")
    count, dimensions = map(int, requested_shape)
    if count < 1 or dimensions not in (2, 3):
        raise ValueError(f"Unsupported dataset kpt_shape={requested_shape!r}")
    if inferred_count and inferred_count != count:
        raise ValueError(
            f"COCO JSON has {inferred_count} keypoints but dataset kpt_shape requests {count}"
        )
    return count, dimensions


def load_coco_pose_index(
    annotation_path: Path,
    root: Path,
    image_roots: Sequence[Path],
    available_images: Sequence[Path],
    class_names: Optional[Sequence[str]],
    requested_shape,
):
    """Return image paths, annotation records, names, category mapping and keypoint shape."""
    with annotation_path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise ValueError(f"Not a COCO dataset JSON (expected images/annotations/categories): {annotation_path}")

    categories = payload["categories"]
    annotations = payload["annotations"]
    number_of_keypoints, keypoint_dimensions = _coco_keypoint_shape(
        categories, annotations, requested_shape
    )
    names, category_to_class = _category_map(categories, class_names)

    images_by_name: Dict[str, List[Path]] = {}
    for image_path in available_images:
        images_by_name.setdefault(image_path.name.casefold(), []).append(image_path.resolve())
    resolved_roots = [root.resolve(), *(path.resolve() for path in image_roots)]
    image_paths: List[Path] = []
    annotations_by_path: Dict[str, List[dict]] = {}
    image_metadata_by_path: Dict[str, dict] = {}
    image_id_to_path: Dict[Any, Path] = {}
    for image_record in payload["images"]:
        file_name = image_record.get("file_name")
        if not file_name:
            raise ValueError(f"COCO image record is missing file_name: {annotation_path}")
        file_path = Path(str(file_name)).expanduser()
        candidates = [file_path] if file_path.is_absolute() else [base / file_path for base in resolved_roots]
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if resolved is None:
            matches = images_by_name.get(file_path.name.casefold(), [])
            if len(matches) == 1:
                resolved = matches[0]
        if resolved is None:
            raise FileNotFoundError(
                f"COCO image {file_name!r} was not found under dataset root or configured image folders"
            )
        image_paths.append(resolved)
        image_id_to_path[image_record["id"]] = resolved
        annotations_by_path[str(resolved)] = []
        image_metadata_by_path[str(resolved)] = {
            "id": int(image_record["id"]),
            "width": int(image_record.get("width", 0)),
            "height": int(image_record.get("height", 0)),
        }

    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        image_path = image_id_to_path.get(annotation.get("image_id"))
        category_id = annotation.get("category_id")
        if image_path is None or category_id not in category_to_class:
            continue
        annotations_by_path[str(image_path)].append(annotation)

    return (
        image_paths,
        annotations_by_path,
        names,
        category_to_class,
        (number_of_keypoints, keypoint_dimensions),
        image_metadata_by_path,
    )


def annotation_source_for_split(data_config: Mapping[str, Any], split: str):
    """Read explicit split annotations from train_annotations/annotations mappings."""
    value = data_config.get(f"{split}_annotations")
    if value is not None:
        return value
    annotations = data_config.get("annotations")
    if isinstance(annotations, Mapping):
        return annotations.get(split)
    return None


__all__ = ["annotation_source_for_split", "load_coco_pose_index"]
