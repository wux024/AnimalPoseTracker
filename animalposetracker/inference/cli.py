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
from pathlib import Path


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
    parser = argparse.ArgumentParser(
        prog="animalpose-cli infer",
        description="Run live pose inference on a camera, video file, or stream URL.",
    )
    parser.add_argument("--weights", required=True,
                        help="Exported model artifact (.onnx, .engine, .mlpackage, ...)")
    parser.add_argument("--config",
                        help="Dataset or project YAML providing class names, skeleton and kpt_shape")
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
    parser.add_argument("--show", action="store_true",
                        help="Display the rendered stream in a window")
    parser.add_argument("--output", help="Write the rendered stream to this video file")
    parser.add_argument("--records", help="Write per-frame pose/tracking records to this JSON file")
    parser.add_argument("--max-frames", type=int,
                        help="Stop after this many frames (useful for smoke tests)")
    return parser


def _resolve_num_classes(engine, kpt_shape, default: int = 1) -> int:
    """Infer the class count from the runtime output tensor width.

    The exported YOLO pose head emits ``4 + nc + K * dims`` channels per
    anchor, so ``nc`` falls out of the runtime shape without extra metadata.
    """
    per_anchor = int(kpt_shape[0]) * int(kpt_shape[1])
    shapes = getattr(engine, "_runtime_output_shapes", {}) or {}
    for shape in shapes.values():
        dims = [int(d) for d in shape]
        if len(dims) >= 3 and dims[1] > 4 + per_anchor:
            return dims[1] - 4 - per_anchor
    return default


def _load_dataset_config(config_arg):
    import yaml

    names, skeleton, kpt_shape = [], [], None
    if config_arg is None:
        return names, skeleton, kpt_shape
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
    return names, skeleton, kpt_shape


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


def _render(frame, detections, skeleton, show_bbox=True):
    import cv2

    rendered = frame.copy()
    for detection in detections:
        keypoints = detection.get("keypoints") or []
        label = detection.get("class_name") or str(detection.get("class_id", 0))
        score = detection.get("score")
        if score is not None:
            label = f"{label} {float(score):.2f}"
        track_id = detection.get("track_id")
        if track_id is not None:
            label = f"#{track_id} {label}"
        bbox = detection.get("bbox_xyxy")
        if show_bbox and bbox:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (60, 180, 75), 2)
            cv2.putText(rendered, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 75), 2, cv2.LINE_AA)
        points = []
        for point in keypoints:
            if len(point) >= 3 and float(point[2]) <= 0.0:
                points.append(None)
                continue
            points.append((int(round(point[0])), int(round(point[1]))))
        for start, end in skeleton:
            if start < len(points) and end < len(points) and points[start] and points[end]:
                cv2.line(rendered, points[start], points[end], (220, 160, 60), 2, cv2.LINE_AA)
        for point in points:
            if point:
                cv2.circle(rendered, point, 4, (80, 80, 230), -1, cv2.LINE_AA)
    return rendered


def _print_infer_banner(rows) -> None:
    if not sys.stdout.isatty():
        return
    from animalposetracker.training.console import environment_line, format_summary_block

    print()
    print(environment_line())
    print()
    print(format_summary_block(rows))
    print()


def _progress(text: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\r" + text.ljust(110))
        sys.stdout.flush()


def _end_progress() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()


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
        RTMPOSE_SIMCC,
        YOLO_POSE_RAW,
        PoseOutputConfig,
    )

    metadata = read_artifact_metadata(weights_path)
    artifact_format = detect_artifact_format(weights_path, metadata)
    engine_key = args.engine or _FORMAT_TO_ENGINE.get(artifact_format, "onnx")
    class_names, skeleton, kpt_shape = _load_dataset_config(args.config)
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
        output_config = PoseOutputConfig(
            schema=RTMPOSE_SIMCC,
            num_keypoints=kpt_shape[0],
            keypoint_dims=kpt_shape[1],
            confidence_threshold=args.conf,
            simcc_split_ratio=float(metadata.get("simcc_split_ratio", 2.0)),
        )
        input_config = StreamInputConfig(
            input_size=input_size,
            color_order=str(metadata.get("color_order", "RGB")),
            pixel_mean=tuple(metadata.get("pixel_mean", (123.675, 116.28, 103.53))),
            pixel_std=tuple(metadata.get("pixel_std", (58.395, 57.12, 57.375))),
            bbox_padding=float(metadata.get("bbox_padding", 1.25)),
        )
        pipeline = StreamInferencePipeline(
            engine,
            output_config,
            input_mode="topdown",
            input_config=input_config,
            tracker=args.tracker,
            class_names=class_names,
        )
    else:
        output_config = PoseOutputConfig(
            schema=YOLO_POSE_RAW,
            num_classes=_resolve_num_classes(engine, kpt_shape),
            num_keypoints=kpt_shape[0],
            keypoint_dims=kpt_shape[1],
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
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
        ("Source", source_text),
        ("Thresholds", f"conf={args.conf} iou={args.iou}"),
        ("Tracking", tracking_desc),
        ("Output", str(args.output) if args.output else "off"),
    ])

    records = []
    frame_index = 0
    started_at = time.time()
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
                rendered = _render(frame, detections, skeleton)
                if writer is not None:
                    writer.write(rendered)
                if args.show:
                    cv2.imshow("animalpose-cli infer", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            elapsed = max(time.time() - started_at, 1e-6)
            _progress(
                f"frame {frame_index + 1} · {len(detections)} poses · "
                f"{(frame_index + 1) / elapsed:.1f} fps"
            )
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    _end_progress()

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
