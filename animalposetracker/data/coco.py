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
    class_names: Optional[Sequence[str]],
    requested_shape,
    image_dir: Optional[Path] = None,
):
    """Return image paths, annotation records, names, category mapping and keypoint shape.

    ``image_dir`` is the split's image folder from the dataset descriptor (the
    ``train:`` / ``val:`` / ``test:`` entry). A bare ``file_name`` is joined onto it,
    which is what the annotator writes; a name with folder parts is relative to the
    dataset root, which is what external COCO files carry.
    """
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

    resolved_root = root.resolve()
    resolved_image_dir = image_dir.resolve() if image_dir is not None else None
    image_paths: List[Path] = []
    annotations_by_path: Dict[str, List[dict]] = {}
    image_metadata_by_path: Dict[str, dict] = {}
    image_id_to_path: Dict[Any, Path] = {}
    for image_record in payload["images"]:
        file_name = image_record.get("file_name")
        if not file_name:
            raise ValueError(f"COCO image record is missing file_name: {annotation_path}")
        # Two accepted forms, both deterministic: a bare name joins the split's image
        # folder (what the annotator writes), a name with folder parts is relative to
        # the dataset root (what external COCO files carry). An absolute path is a
        # broken dataset -- report it instead of searching other roots or guessing by
        # filename, which used to hide layout mistakes until the metrics looked wrong.
        file_path = Path(str(file_name))
        if file_path.is_absolute():
            raise ValueError(
                f"COCO image file_name must be relative to the dataset root {root}; "
                f"got an absolute path: {file_name!r}"
            )
        if len(file_path.parts) == 1 and resolved_image_dir is not None:
            resolved = (resolved_image_dir / file_path).resolve()
        else:
            resolved = (resolved_root / file_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"COCO image {file_name!r} does not exist under the dataset root {root}"
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
