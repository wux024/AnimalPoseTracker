"""Live stream inference worker for cameras, video files, and stream URLs.

This CLI drives the engine-based runtime (ONNX/OpenCV/OpenVINO/TensorRT/
CoreML/CANN) through ``InferenceEngine`` and decodes outputs with
``StreamInferencePipeline``. Tracking and temporal filtering reuse the same
building blocks as ``predict --tracker --pose-filter``.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

from tqdm import tqdm


_ENGINE_NAMES = {
    "onnx": "ONNX",
    "opencv": "OpenCV",
    "openvino": "OpenVINO",
    "tensorrt": "TensorRT",
    "coreml": "CoreML",
    "cann": "CANN",
}

_FORMAT_TO_ENGINE = {
    "onnx": "onnx",
    "engine": "tensorrt",
    "coreml": "coreml",
    "openvino": "openvino",
}


def _make_argument_parser() -> argparse.ArgumentParser:
    # Imported lazily: this package loads its backends on demand (see
    # inference/__init__.py), and the decoder module pulls in cv2/numpy.
    from animalposetracker.inference.schema_resolution import SUPPORTED_YOLO_SCHEMAS

    parser = argparse.ArgumentParser(
        prog="animalpose-cli infer",
        description="Run live pose inference on a camera, video file, or stream URL.",
    )
    parser.add_argument("--weights", required=True,
                        help="Exported model artifact (.onnx, .engine, .mlpackage, ...)")
    parser.add_argument("--config",
                        help="Dataset or project YAML providing class names, skeleton and kpt_shape")
    parser.add_argument("--schema", default="auto", choices=("auto",) + SUPPORTED_YOLO_SCHEMAS,
                        help="Pose output schema of the artifact; 'auto' resolves it from the "
                             "config yaml, sidecar metadata, then the runtime output shapes")
    parser.add_argument("--engine", choices=sorted(_ENGINE_NAMES),
                        help="Inference backend; defaults to the artifact format")
    parser.add_argument("--device", default="CPU", help="Backend device string (default: CPU)")
    parser.add_argument("--bits", default="FP32", choices=("FP32", "FP16", "INT8"),
                        help="Model weight precision used by the backend")
    parser.add_argument("--source", required=True,
                        help="Camera index (0, 1, ...), video file path, or stream URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Fallback square input size when no sidecar metadata exists")
    parser.add_argument("--tracker",
                        help="Track detections with this AnimalPoseTracker algorithm")
    parser.add_argument("--pose-filter", dest="pose_filter",
                        choices=("none", "median", "savgol"), default="none",
                        help="Smooth tracked records after the stream ends")
    parser.add_argument("--filter-window", type=int, help="Temporal filter window length")
    parser.add_argument("--filter-polyorder", type=int, help="Savitzky-Golay polynomial order")
    parser.add_argument("--filter-confidence", type=float,
                        help="Keypoint confidence threshold for temporal filtering")
    parser.add_argument("--filter-max-gap", type=int,
                        help="Maximum missing-frame gap to interpolate when filtering")
    parser.add_argument("--agnostic-nms", dest="agnostic_nms", action="store_true",
                        help="Use class-agnostic NMS")
    parser.add_argument("--class-aware-nms", dest="agnostic_nms", action="store_false",
                        help="Apply NMS separately for each class")
    parser.set_defaults(agnostic_nms=True)
    parser.add_argument("--show-boxes", dest="show_boxes", action="store_true")
    parser.add_argument("--no-show-boxes", dest="show_boxes", action="store_false")
    parser.add_argument("--show-labels", dest="show_labels", action="store_true")
    parser.add_argument("--no-show-labels", dest="show_labels", action="store_false")
    parser.add_argument("--show-keypoints", dest="show_keypoints", action="store_true")
    parser.add_argument("--no-show-keypoints", dest="show_keypoints", action="store_false")
    parser.add_argument("--show-skeletons", dest="show_skeletons", action="store_true")
    parser.add_argument("--no-show-skeletons", dest="show_skeletons", action="store_false")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Rendered bounding box and skeleton line width")
    parser.add_argument("--point-radius", "--kpt-radius", dest="point_radius", type=int, default=4,
                        help="Rendered keypoint radius")
    parser.set_defaults(show_boxes=True, show_labels=True, show_keypoints=True, show_skeletons=True)
    parser.add_argument("--show", action="store_true",
                        help="Display the rendered stream in a window")
    parser.add_argument("--output", help="Write the rendered stream to this video file")
    parser.add_argument("--records", help="Write per-frame pose/tracking records to this JSON file")
    parser.add_argument("--max-frames", type=int,
                        help="Stop after this many frames (useful for smoke tests)")
    return parser


def _resolve_num_classes(engine, kpt_shape, default: int = 1, schema: str = None) -> int:
    """Infer the class count from the runtime output tensor width.

    Valid for :data:`~animalposetracker.postprocessing.output_decoders.YOLO_POSE_RAW`
    only: that layout emits ``4 + nc + K * dims`` channels per anchor, so ``nc``
    falls out of the runtime shape without extra metadata.

    Post-NMS and end-to-end layouts have a fixed ``6 + K * dims`` channel count and
    put the *detection count* on the axis read here, so the arithmetic would return
    a meaningless number -- 300 detections over 17x3 keypoints would report 245
    classes. Callers must pass ``schema`` so those layouts take ``nc`` from the
    configured class names instead; ``schema=None`` keeps the historical behaviour
    for callers that predate the parameter.
    """
    if schema is not None:
        from animalposetracker.postprocessing.output_decoders import YOLO_POSE_RAW

        if schema != YOLO_POSE_RAW:
            warnings.warn(
                f"Slot width does not encode the class count for schema {schema!r}; "
                "keeping the supplied default instead of deriving one.",
                stacklevel=2,
            )
            return default
    per_anchor = int(kpt_shape[0]) * int(kpt_shape[1])
    shapes = getattr(engine, "_runtime_output_shapes", {}) or {}
    for shape in shapes.values():
        dims = [int(d) for d in shape]
        if len(dims) >= 3 and dims[1] > 4 + per_anchor:
            return dims[1] - 4 - per_anchor
    return default


def _load_dataset_config(config_arg):
    """Read the infer-relevant keys from a dataset or project yaml.

    Returns ``(names, skeleton, kpt_shape, schema)``; ``schema`` is ``None`` when
    the file does not declare one, which lets the caller fall through to the
    artifact metadata and then to shape inference.
    """
    import yaml

    names, skeleton, kpt_shape, schema = [], [], None, None
    if config_arg is None:
        return names, skeleton, kpt_shape, schema
    config_path = Path(config_arg).expanduser().resolve()
    mapping = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_names = mapping.get("names") or {}
    if isinstance(raw_names, dict):
        names = [str(raw_names[key]) for key in sorted(raw_names)]
    else:
        names = [str(item) for item in raw_names]
    skeleton = [list(map(int, pair)) for pair in (mapping.get("skeleton") or [])]
    if mapping.get("kpt_shape"):
        kpt_shape = tuple(map(int, mapping["kpt_shape"]))
    raw_schema = mapping.get("schema") or mapping.get("output_schema")
    if raw_schema:
        schema = str(raw_schema).strip().lower()
    return names, skeleton, kpt_shape, schema


def _instance_to_record(instance, class_names) -> dict:
    keypoints = instance.keypoints_xy
    scores = instance.keypoint_scores
    if scores is None:
        scores = instance.keypoints_visible
    if scores is not None:
        import numpy as np

        scores_col = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
        if scores_col.shape[0] == keypoints.shape[0]:
            keypoints = np.concatenate([keypoints, scores_col], axis=1)
    class_id = int(instance.class_id) if instance.class_id is not None else 0
    return {
        "bbox_xyxy": None if instance.bbox_xyxy is None else [
            float(v) for v in instance.bbox_xyxy
        ],
        "score": float(instance.score) if instance.score is not None else 1.0,
        "class_id": class_id,
        "class_name": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
        "keypoints": keypoints.tolist(),
    }


def _pixel_points(keypoints):
    """Integer pixel coordinate per keypoint, ``None`` where not scorable.

    A keypoint whose score is not positive carries no position, so it is
    neither drawn nor connected to an edge.
    """
    points = []
    for point in keypoints:
        if len(point) >= 3 and float(point[2]) <= 0.0:
            points.append(None)
            continue
        points.append((int(round(point[0])), int(round(point[1]))))
    return points


def _detection_label(detection) -> str:
    """``#track class score`` label for one detection, in that order."""
    label = detection.get("class_name") or str(detection.get("class_id", 0))
    score = detection.get("score")
    if score is not None:
        label = f"{label} {float(score):.2f}"
    track_id = detection.get("track_id")
    if track_id is not None:
        label = f"#{track_id} {label}"
    return label


def _render(
    frame,
    detections,
    skeleton,
    *,
    show_bbox=True,
    show_labels=True,
    show_keypoints=True,
    show_skeletons=True,
    line_width=2,
    point_radius=4,
):
    # Imported here rather than at module scope: this package loads its drawing
    # dependency on demand, so plain `infer --help` stays cheap.
    from animalposetracker.rendering import INFERENCE_STYLE, draw_instance, pixel_box

    rendered = frame.copy()
    for detection in detections:
        draw_instance(
            rendered,
            pixel_box(detection.get("bbox_xyxy")),
            _detection_label(detection),
            _pixel_points(detection.get("keypoints") or []),
            skeleton,
            style=INFERENCE_STYLE,
            show_box=show_bbox,
            show_label=show_labels,
            show_keypoints=show_keypoints,
            show_skeleton=show_skeletons,
            line_width=line_width,
            point_radius=point_radius,
        )
    return rendered


def _print_infer_banner(rows) -> None:
    if not sys.stdout.isatty():
        return
    from animalposetracker.console import environment_line, format_summary_block

    print()
    print(environment_line())
    print()
    print(format_summary_block(rows))
    print()


def run(argv=None) -> int:
    args = _make_argument_parser().parse_args(argv)
    weights_path = Path(args.weights).expanduser().resolve()
    if weights_path.suffix.lower() == ".pt":
        raise ValueError(
            "infer runs exported artifacts; export first (animalpose-cli export) "
            "or use predict for .pt checkpoints"
        )

    import cv2

    from animalposetracker.artifacts import detect_artifact_format, read_artifact_metadata
    from animalposetracker.inference.inferencer import InferenceEngine
    from animalposetracker.inference.stream_inference import (
        StreamInferencePipeline,
        StreamInputConfig,
    )
    from animalposetracker.postprocessing.output_decoders import (
        YOLO_POSE_RAW,
        PoseOutputConfig,
    )
    from animalposetracker.inference.schema_resolution import resolve_output_schema

    metadata = read_artifact_metadata(weights_path)
    artifact_format = detect_artifact_format(weights_path, metadata)
    engine_key = args.engine or _FORMAT_TO_ENGINE.get(artifact_format, "onnx")
    class_names, skeleton, kpt_shape, config_schema = _load_dataset_config(args.config)
    if kpt_shape is None and metadata.get("kpt_shape"):
        kpt_shape = tuple(map(int, metadata["kpt_shape"]))
    if kpt_shape is None:
        raise ValueError(
            "kpt_shape is unknown: pass --config with a dataset yaml or use an "
            "artifact exported with sidecar metadata"
        )
    input_size = tuple(map(int, metadata.get("input_size") or (args.imgsz, args.imgsz)))

    engine = InferenceEngine(
        config=args.config,
        weights_path=str(weights_path),
        input_width=input_size[0],
        input_height=input_size[1],
        engine=_ENGINE_NAMES[engine_key],
        device=args.device,
        model_bits=args.bits,
        conf=args.conf,
        iou=args.iou,
        output_names=tuple(metadata.get("outputs") or ()),
    )
    engine.model_init()

    head = str(metadata.get("head") or "")
    if not head:
        head = "SimCCHead" if "simcc_x" in tuple(metadata.get("outputs") or ()) else "YOLOPoseHead"
    if head == "SimCCHead":
        raise ValueError(
            f"{weights_path} holds a SimCCHead (AnimalViTPose) model, and the "
            "`infer` command cannot run one: top-down decoding needs a separate "
            "detector, which this command has no way to supply. Run it through "
            "`animalposetracker predict` instead."
        )

    # A YOLOPoseHead artifact can emit any of the three ultralytics layouts.
    # The CLI flag wins over the config yaml, which wins over the sidecar,
    # which wins over the runtime shape; see schema_resolution for the rules.
    schema, inferred_num_classes = resolve_output_schema(
        explicit_schema=args.schema if args.schema != "auto" else config_schema,
        metadata=metadata,
        output_shapes=getattr(engine, "_runtime_output_shapes", {}) or {},
        num_keypoints=kpt_shape[0],
        keypoint_dims=kpt_shape[1],
    )
    if inferred_num_classes is not None:
        num_classes = inferred_num_classes
    elif schema == YOLO_POSE_RAW:
        num_classes = _resolve_num_classes(engine, kpt_shape, schema=schema)
    else:
        # Post-NMS and end-to-end rows carry the class id directly, so the
        # only thing num_classes feeds is the class-name lookup in the
        # rendered records. The configured names are the right source.
        num_classes = max(1, len(class_names))
    output_config = PoseOutputConfig(
        schema=schema,
        num_classes=num_classes,
        num_keypoints=kpt_shape[0],
        keypoint_dims=kpt_shape[1],
        # The decoder looks tensors up by name, and the engine already asked the
        # backend which names exist. Handing them over keeps the two in step:
        # a foreign artifact names its output "output0" (ultralytics) rather
        # than the "predictions" default, so without this an artifact that
        # carries no sidecar could not be decoded at all. An empty tuple here
        # falls back to the schema default, exactly as before.
        output_names=tuple(engine.output_names or ()),
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        class_agnostic_nms=args.agnostic_nms,
    )
    input_config = StreamInputConfig(
        input_size=input_size,
        preprocess_mode="engine",
        color_order=str(metadata.get("color_order", "RGB")),
    )
    pipeline = StreamInferencePipeline(
        engine,
        output_config,
        input_mode="full_frame",
        input_config=input_config,
        tracker=args.tracker,
        class_names=class_names,
    )

    source_text = str(args.source)
    capture_source = int(source_text) if source_text.isdigit() else source_text
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open inference source: {source_text}")
    frame_total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    if args.max_frames:
        frame_total = min(frame_total, args.max_frames) if frame_total else args.max_frames

    writer = None
    if args.output:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video {output_path}")

    tracking_desc = str(args.tracker) if args.tracker else "off"
    if args.tracker and args.pose_filter != "none":
        tracking_desc += f" + {args.pose_filter} filter"
    _print_infer_banner([
        ("Weights", str(weights_path)),
        ("Engine", f"{_ENGINE_NAMES[engine_key]} ({args.bits}, {args.device})"),
        ("Head", head),
        # Surfaced because 'auto' resolution can settle on a schema the caller did
        # not state, and the banner is the only place that decision becomes visible.
        ("Schema", output_config.schema),
        ("Source", source_text),
        ("Thresholds", f"conf={args.conf} iou={args.iou}"),
        ("Tracking", tracking_desc),
        ("Output", str(args.output) if args.output else "off"),
    ])

    records = []
    frame_index = 0
    started_at = time.time()
    frame_progress = tqdm(
        total=frame_total or None,
        desc="infer",
        unit="frame",
        dynamic_ncols=True,
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.tracker:
                detections = pipeline.track_frame(frame, frame_index)
            else:
                instances = pipeline.process_frame(frame)
                detections = [_instance_to_record(instance, class_names) for instance in instances]
            records.append({"frame": frame_index, "predictions": detections})
            if writer is not None or args.show:
                rendered = _render(
                    frame,
                    detections,
                    skeleton,
                    show_bbox=args.show_boxes,
                    show_labels=args.show_labels,
                    show_keypoints=args.show_keypoints,
                    show_skeletons=args.show_skeletons,
                    line_width=args.line_width,
                    point_radius=args.point_radius,
                )
                if writer is not None:
                    writer.write(rendered)
                if args.show:
                    cv2.imshow("animalpose-cli infer", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            elapsed = max(time.time() - started_at, 1e-6)
            current_frame = frame_index + 1
            frame_progress.set_postfix(
                poses=len(detections), fps=f"{current_frame / elapsed:.1f}", refresh=False
            )
            frame_progress.update(1)
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        frame_progress.close()

    if args.records:
        payload = records
        if args.pose_filter != "none":
            if not args.tracker:
                raise ValueError("--pose-filter requires --tracker so records carry track IDs")
            from animalposetracker.postprocessing.workflows import filter_tracked_video_records

            payload = filter_tracked_video_records(
                records,
                method=args.pose_filter,
                window_length=args.filter_window or 5,
                polyorder=args.filter_polyorder or 2,
                confidence_threshold=(
                    0.25 if args.filter_confidence is None else args.filter_confidence
                ),
                max_gap=0 if args.filter_max_gap is None else args.filter_max_gap,
            )
        records_path = Path(args.records).expanduser().resolve()
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    elapsed = max(time.time() - started_at, 1e-6)
    print(
        f"Inference complete: {frame_index} frames in {elapsed:.1f}s "
        f"({frame_index / elapsed:.1f} fps)"
    )
    return 0
