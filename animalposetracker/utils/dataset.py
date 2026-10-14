import json
from pathlib import Path
from functools import wraps
from tqdm import tqdm

def with_progress_bar(desc=None):
    """Decorator to add progress bar to functions that process files"""
    def decorator(func):
        @wraps(func)
        def wrapper(labels_path, *args, **kwargs):
            # Handle directory input (multiple files)
            if Path(labels_path).is_dir():
                files = [f for f in Path(labels_path).iterdir() 
                        if f.is_file() and f.suffix == ".json"]
                if not files:
                    raise FileNotFoundError(f"No JSON files found in {labels_path}")
                
                # Initialize progress bar for multiple files
                with tqdm(files, desc=desc or f"Processing {func.__name__}") as pbar:
                    result = func(labels_path, *args, **kwargs)  # Process first file
                    pbar.update(len(files))  # Ensure progress completes
                return result
            
            # Handle single file input
            else:
                with tqdm(total=1, desc=desc or f"Processing {func.__name__}") as pbar:
                    result = func(labels_path, *args, **kwargs)  # Process file
                    pbar.update(1)
                return result
        return wrapper
    return decorator

def convert_labels(labels_path, project_config, output_dir=None, output_format='coco'):
    """
    Convert labels in a JSON file to a specified format.
    """
    if output_format == 'coco':
        convert_labels_to_coco(labels_path, project_config, output_dir)
    elif output_format == 'yolo':
        convert_labels_to_yolo(labels_path, output_dir)
    elif output_format == 'all':
        convert_labels_to_coco(labels_path, project_config, output_dir)
        convert_labels_to_yolo(labels_path, output_dir)
    else:
        raise ValueError(f'Unsupported output format: {output_format}')

def _has_bounding_box(bbox) -> bool:
    """Whether a label carries a usable box.

    The annotator writes ``bbox: None`` for targets that only ever got keypoints drawn on them,
    and both converters below need four numbers to place an instance.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    return all(isinstance(value, (int, float)) for value in bbox)


@with_progress_bar(desc="Converting to COCO format")
def convert_labels_to_coco(labels_path, project_config, output_dir=None):
    """
    Convert labels in a JSON file to COCO format.
    
    Args:
        labels_path (str/Path): Path to directory containing JSON label files
        output_dir (str/Path, optional): Output directory. Defaults to same as input.
    
    Returns:
        Path: Path to the generated COCO format JSON file
    """
    if output_dir is None:
        output_dir = Path(labels_path) / "coco_format" / "coco_format.json"
    else:
        output_dir = Path(output_dir) / "coco_format.json"
    # Create output directory
    output_dir.parent.mkdir(exist_ok=True, parents=True)
    # Initialize COCO format dictionary
    coco = {
        "info": {}, 
        "licenses": "", 
        "images": [], 
        "annotations": [], 
        "categories": []
    }
    
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # ID counters and mappings
    image_id_counter = 1
    annotation_id_counter = 1
    skipped_annotations = 0
    
    for json_file in labels_path.iterdir():
        if not json_file.is_file() or not json_file.suffix == ".json":
            continue
        
        with open(json_file, "r") as f:
            data = json.load(f)
            
            # Process images
            for image in data.get("images", []):
                # Create new image entry with updated ID
                new_image = {
                    "id": image_id_counter,
                    "file_name": image.get("file_name", ""),
                    "width": image.get("width", 0),
                    "height": image.get("height", 0),
                }
                coco["images"].append(new_image)
                
                # Process annotations for this image
                for ann in data.get("annotations", []):
                    if ann["image_id"] != image["id"]:
                        continue
                    if ann.get("category_id") is None or not _has_bounding_box(ann.get("bbox")):
                        # Point-only targets have no category to add 1 to and no box to write;
                        # counting them keeps the loss visible instead of failing the export.
                        skipped_annotations += 1
                        continue

                    # Create new annotation with updated IDs
                    new_ann = {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": ann["category_id"] + 1,
                        "area": ann.get("area", 0),
                        "bbox": ann.get("bbox", []),
                        "keypoints": ann.get("keypoints", []),
                        "iscrowd": ann.get("iscrowd", 0)
                    }
                    coco["annotations"].append(new_ann)
                    annotation_id_counter += 1
                
                image_id_counter += 1
    
    if skipped_annotations:
        print(
            f"Skipped {skipped_annotations} annotation(s) with no bounding box "
            "while converting to COCO format."
        )
    # The shape's home is dataset.yaml (``names`` as a mapping) while the
    # historical project spelling carried the three lists; accept both.
    class_names = project_config.get("classes_name")
    if class_names is None:
        names = project_config.get("names")
        if isinstance(names, dict):
            class_names = [str(value) for _, value in sorted(names.items())]
        else:
            class_names = list(names or [])
    keypoints_name = project_config.get("keypoints_name")
    if not keypoints_name:
        shape = project_config.get("kpt_shape") or [0]
        keypoints_name = [f"kpt_{index}" for index in range(int(shape[0]))]
    skeleton = project_config.get("skeleton") or []
    coco["categories"] = [
        {
            "supercategory": "",
            "id": i+1,
            "name": name,
            "keypoints": keypoints_name,
            "skeleton": skeleton
        } for i, name in enumerate(class_names)
    ]
    
    # Write output file
    with open(output_dir, "w") as f:
        json.dump(coco, f, indent=4)
    
    return output_dir

@with_progress_bar(desc="Converting to COCO format")      
def convert_labels_to_yolo(labels_path, output_dir=None):
    """
    Convert labels in a JSON file to YOLO format.
    
    Args:
        labels_path (str/Path): Path to directory containing single-image JSON label files
        output_dir (str/Path, optional): Output directory. Defaults to same as input.
    
    Returns:
        Path: Path to the output directory containing YOLO format files
    """
    if output_dir is None:
        output_dir = Path(labels_path) / "yolo_format"
    else:
        output_dir = Path(output_dir) / "yolo_format"
    
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for json_file in labels_path.iterdir():
        if not json_file.is_file() or not json_file.suffix == ".json":
            continue
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Verify JSON structure contains exactly one image
        if "images" not in data or len(data["images"]) != 1:
            print(f"Skipping {json_file.name}: does not contain exactly one image")
            continue
        
        # Get image info
        img = data["images"][0]
        img_width = img["width"]
        img_height = img["height"]
        filename = Path(img["file_name"]).stem
        
        # Prepare YOLO content
        yolo_lines = []
        skipped_annotations = 0
        for ann in data.get("annotations", []):
            if ann.get("category_id") is None or not _has_bounding_box(ann.get("bbox")):
                skipped_annotations += 1
                continue
            # Convert bbox from [x,y,width,height] to YOLO format [center_x, center_y, width, height] (normalized)
            x, y, w, h = ann["bbox"]
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            
            # Process keypoints if they exist
            yolo_keypoints = []
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                for i in range(0, len(keypoints), 3):
                    kx = keypoints[i] / img_width
                    ky = keypoints[i+1] / img_height
                    kv = keypoints[i+2]  # visibility flag
                    yolo_keypoints.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
            
            # Format: class_id center_x center_y width height [keypoints...]
            line = f"{ann['category_id']} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
            if yolo_keypoints:
                line += " " + " ".join(yolo_keypoints)
            yolo_lines.append(line)
        
        # Write to TXT file (same name as image)
        output_path = output_dir / f"{filename}.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(yolo_lines))
        if skipped_annotations:
            print(
                f"Skipped {skipped_annotations} annotation(s) with no bounding box "
                f"in {json_file.name} while converting to YOLO format."
            )

    return output_dir