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
from collections import deque
from pathlib import Path

import yaml
from tqdm import tqdm

from animalposetracker.inference.session import (
    ENGINE_NAMES,
    FORMAT_TO_ENGINE,
    build_inference_session,
    instance_to_record,
    load_inference_config,
    pixel_points,
    resolve_num_classes,
)
from animalposetracker.postprocessing.nms_params import (
    DEPLOY_DETECTOR_CONF,
    DEPLOY_NMS_AGNOSTIC,
    DEPLOY_NMS_CONF,
    DEPLOY_NMS_IOU,
)


# The underscore names stay: they are what this module has always exposed, and the tests reach
# for them directly. The implementations now live in inference/session.py, which the GUI's
# inferencer page shares, so the CLI and the page cannot drift apart.
_ENGINE_NAMES = ENGINE_NAMES
_FORMAT_TO_ENGINE = FORMAT_TO_ENGINE
_resolve_num_classes = resolve_num_classes
_load_dataset_config = load_inference_config
_instance_to_record = instance_to_record
_pixel_points = pixel_points


# The run-time keys an infer configuration file may carry, in the canonical
# spellings (the parser's dest names). Data-description keys ride along in the
# same file for the session to read and are skipped here.
_INFER_FILE_KEYS = frozenset({
    "poser_weights", "detector_weights", "engine", "device", "bits", "imgsz",
    "source", "conf", "iou", "agnostic_nms", "detector_conf",
    "detector_box_format", "detector_coord_mode", "detector_preprocess_mode",
    "schema", "show_boxes", "show_labels", "show_keypoints", "show_skeletons",
    "line_width", "point_radius", "tracker", "pose_filter",
    "filter_window", "filter_polyorder", "filter_confidence", "filter_max_gap",
    "output", "records", "show", "max_frames",
})
_DATA_DESCRIPTION_KEYS = frozenset({
    "names", "classes_name", "keypoints", "skeleton",
    "kpt_shape", "visible", "output_schema",
    # dataset-style keys, for a file borrowed from a dataset.yaml
    "path", "train", "val", "test", "annotation_format", "flip_idx",
    "kpt_oks_sigmas", "keypoint_names", "upper_body_ids", "lower_body_ids",
})


def _infer_file_values(config_path) -> dict:
    """Run-time values from an inference file; unknown keys warn.

    The inferencer reads no project configuration -- this file (any path the
    operator points ``--config`` at, template: ``--write-template``) is a
    plain two-layer default set: the command line beats it, it beats the
    built-in defaults. Data-description keys stay untouched here; the
    session reads them from the same file.
    """
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Inference config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    values = {}
    for key, value in mapping.items():
        key = str(key)
        if key in _INFER_FILE_KEYS:
            values[key] = value
        elif key not in _DATA_DESCRIPTION_KEYS:
            print(f"warning: unknown key {key!r} in {path}; ignored", file=sys.stderr)
    return values


def _write_template(destination) -> int:
    """Copy the packaged infer template out (or print it with ``-``)."""
    from animalposetracker.cfg import CFG_DIR

    template_path = CFG_DIR / "infer.example.yaml"
    content = template_path.read_text(encoding="utf-8")
    if str(destination) == "-":
        print(content, end="")
        return 0
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"Template written: {target}")
    return 0


def _make_argument_parser() -> argparse.ArgumentParser:
    # Imported lazily: this package loads its backends on demand (see
    # inference/__init__.py), and the decoder module pulls in cv2/numpy.
    from animalposetracker.inference.schema_resolution import SUPPORTED_YOLO_SCHEMAS

    parser = argparse.ArgumentParser(
        prog="animalpose-cli infer",
        description="Run live pose inference on a camera, video file, or stream URL.",
    )
    parser.add_argument("--poser-weights",
                        help="Exported pose model artifact (.onnx, .engine, .mlpackage, ...); "
                             "may also be set as 'poser_weights' in the --config file")
    parser.add_argument("--detector-weights",
                        help="Detector ONNX that enables the two-stage top-down stream "
                             "(for example an RT-DETR export)")
    parser.add_argument("--detector-conf", type=float, default=DEPLOY_DETECTOR_CONF,
                        help="Detector confidence threshold for top-down streaming")
    parser.add_argument("--detector-box-format", dest="detector_box_format",
                        choices=("cxcywh", "xyxy"), default="cxcywh",
                        help="Row form of the detector boxes: 'cxcywh' (RT-DETR) or "
                             "'xyxy' (the NMS-free YOLO exports)")
    parser.add_argument("--detector-coord-mode", dest="detector_coord_mode",
                        choices=("normalized", "pixels"), default="normalized",
                        help="Detector coordinates: 'normalized' (RT-DETR) or "
                             "'pixels' of the network input (YOLO family)")
    parser.add_argument("--detector-preprocess-mode", dest="detector_preprocess_mode",
                        choices=("stretch", "letterbox"), default="stretch",
                        help="Detector preprocessing: 'stretch' (RT-DETR) or "
                             "'letterbox' (YOLO family)")
    parser.add_argument("--config",
                        help="Infer configuration file: data description (names / kpt_shape / ...) "
                             "and/or run-time defaults (conf / device / ...); the command line wins "
                             "(template: --write-template)")
    parser.add_argument(
        "--write-template", dest="write_template", nargs="?", const="infer.example.yaml",
        metavar="PATH",
        help="Write the infer configuration template (default: ./infer.example.yaml) and exit",
    )
    parser.add_argument("--schema", default="auto", choices=("auto",) + SUPPORTED_YOLO_SCHEMAS,
                        help="Pose output schema of the artifact; 'auto' resolves it from the "
                             "config yaml, sidecar metadata, then the runtime output shapes")
    parser.add_argument("--engine", choices=sorted(_ENGINE_NAMES),
                        help="Inference backend; defaults to the artifact format")
    parser.add_argument("--device", help="Backend device string (e.g. 'AMD CPU'); defaults to the first device the engine offers on this machine's platform")
    parser.add_argument("--bits", default="FP32", choices=("FP32", "FP16", "INT8"),
                        help="Model weight precision used by the backend")
    parser.add_argument("--source",
                        help="Camera index (0, 1, ...), video file path, or stream URL; "
                             "may also be set as 'source' in the --config file")
    parser.add_argument("--conf", type=float, default=DEPLOY_NMS_CONF, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEPLOY_NMS_IOU, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Fallback square input size when no sidecar metadata exists")
    parser.add_argument("--tracker",
                        help="Track detections with this AnimalPoseTracker algorithm")
    # The tracker's own knobs, generated from the dataclass: a new field appears here
    # without a second list to keep in sync. --tracker alone keeps every default.
    import dataclasses

    from animalposetracker.tracking.config import TrackerConfig as _TrackerConfig

    for _item in dataclasses.fields(_TrackerConfig):
        _name = _item.name
        if _name == "algorithm":
            continue
        _flag = f"--track-{_name.replace('_', '-')}"
        _value_type = type(_item.default)
        if _value_type is bool:
            parser.add_argument(_flag, dest=f"track_{_name}", action="store_true", default=None)
            parser.add_argument(f"--no-{_flag[2:]}", dest=f"track_{_name}", action="store_false",
                                default=None)
        elif _value_type in (int, float, str):
            parser.add_argument(_flag, dest=f"track_{_name}", type=_value_type, default=None)
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
    parser.set_defaults(agnostic_nms=DEPLOY_NMS_AGNOSTIC)
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
    parser.add_argument("--point-radius", dest="point_radius", type=int, default=4,
                        help="Rendered keypoint radius")
    parser.add_argument("--kpt-radius", dest="point_radius", type=int, default=4,
                        help="deprecated alias of --point-radius")
    parser.set_defaults(show_boxes=True, show_labels=True, show_keypoints=True, show_skeletons=False)
    parser.add_argument("--show", action="store_true",
                        help="Display the rendered stream in a window")
    parser.add_argument("--output", help="Write the rendered stream to this video file")
    parser.add_argument("--records", help="Write per-frame pose/tracking records to this JSON file")
    parser.add_argument("--max-frames", type=int,
                        help="Stop after this many frames (useful for smoke tests)")
    parser.add_argument("--trail-length", dest="trail_length", type=int, default=0,
                        help="Draw the last N box centres per tracked identity as a trail "
                             "(0 turns trails off; needs --tracker)")
    return parser


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


def _update_trails(trails, detections, limit: int) -> None:
    """Append one bbox-centre point per tracked detection, keeping the newest ``limit``.

    Keyed by track id, so each identity keeps its own polyline. Detections without a
    track id (tracking off) are ignored.
    """
    for detection in detections:
        track_id = detection.get("track_id")
        box = detection.get("bbox_xyxy")
        if track_id is None or box is None:
            continue
        centre = (
            int(round((float(box[0]) + float(box[2])) / 2.0)),
            int(round((float(box[1]) + float(box[3])) / 2.0)),
        )
        history = trails.get(int(track_id))
        if history is None:
            history = deque(maxlen=max(int(limit), 2))
            trails[int(track_id)] = history
        history.append(centre)


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
    trails=None,
):
    # Imported here rather than at module scope: this package loads its drawing
    # dependency on demand, so plain `infer --help` stays cheap.
    from animalposetracker.rendering import INFERENCE_STYLE, draw_instance, pixel_box

    rendered = frame.copy()
    for detection in detections:
        track_id = detection.get("track_id")
        trail = None
        if trails is not None and track_id is not None:
            trail = trails.get(int(track_id))
        draw_instance(
            rendered,
            pixel_box(detection.get("bbox_xyxy")),
            _detection_label(detection),
            _pixel_points(detection.get("keypoints") or []),
            skeleton,
            style=INFERENCE_STYLE,
            track_id=int(track_id) if track_id is not None else None,
            trail=trail,
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
    parser = _make_argument_parser()
    # One probe pass first: --write-template must not demand the required
    # arguments, and an inference file's values become the parser defaults
    # before the real parse -- so explicit flags still win.
    probe, _unknown = parser.parse_known_args(argv)
    if probe.write_template is not None:
        return _write_template(probe.write_template)
    parser.set_defaults(**_infer_file_values(probe.config))
    args = parser.parse_args(argv)
    # `source: 0` is a camera index, not an absence -- check for the empty
    # value, not for falsiness.
    if args.poser_weights is None or not str(args.poser_weights).strip():
        raise ValueError(
            "--poser-weights is required: name the exported artifact, or set "
            "'poser_weights' in the --config file"
        )
    if args.source is None or not str(args.source).strip():
        raise ValueError(
            "--source is required: a camera index, video path or stream URL, "
            "or set 'source' in the --config file"
        )
    weights_path = Path(args.poser_weights).expanduser().resolve()
    if weights_path.suffix.lower() == ".pt":
        raise ValueError(
            "infer runs exported artifacts; export first (animalpose-cli export) "
            "or use predict for .pt checkpoints"
        )

    import cv2

    from animalposetracker.inference.stream_inference import StreamInferencePipeline

    # Artifact loading, schema resolution and the decode contract live in inference/session.py,
    # where the GUI's inferencer page reads them from too.
    session = build_inference_session(
        weights_path=weights_path,
        config_path=args.config,
        engine=args.engine,
        device=args.device,
        bits=args.bits,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        schema=args.schema,
        imgsz=args.imgsz,
        topdown=bool(args.detector_weights),
    )
    engine = session.engine
    output_config = session.output_config
    input_config = session.input_config
    class_names = list(session.class_names)
    skeleton = tuple(session.skeleton)
    head = session.head
    engine_key = session.engine_key
    detector = None
    input_mode = "full_frame"
    if args.detector_weights:
        from animalposetracker.inference.detector_adapter import OnnxDetectorAdapter

        detector = OnnxDetectorAdapter(
            Path(args.detector_weights).expanduser().resolve(),
            conf_threshold=args.detector_conf,
            box_format=args.detector_box_format,
            coord_mode=args.detector_coord_mode,
            preprocess_mode=args.detector_preprocess_mode,
        )
        input_mode = "topdown"
    tracker_config = None
    if args.tracker:
        from animalposetracker.tracking.config import TrackerConfig

        overrides = {}
        for item in dataclasses.fields(TrackerConfig):
            if item.name == "algorithm":
                continue
            value = getattr(args, f"track_{item.name}", None)
            if value is not None:
                overrides[item.name] = value
        # validate() runs here so a bad combination (with_reid without a model, ReID on
        # bytetrack) fails before the first frame instead of inside the stream loop.
        tracker_config = TrackerConfig(algorithm=args.tracker, **overrides).validate()

    pipeline = StreamInferencePipeline(
        engine,
        output_config,
        input_mode=input_mode,
        input_config=input_config,
        detector=detector,
        tracker=tracker_config,
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
        ("Poser weights", str(weights_path)),
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

    # Warm the backend once, before any timing starts: the first real frame would otherwise
    # pay CUDA context creation, kernel compilation and NMS caching, and the stage averages
    # would be reported from a sample that includes that setup.
    pipeline.warmup()

    records = []
    trails = {}
    frame_index = 0
    started_at = time.time()
    stage_keys = ("preprocess", "inference", "postprocess")
    stage_totals = {key: 0.0 for key in stage_keys}
    stage_frames = 0
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
            # Stage timings are measured inside the pipeline, so tracking and plain inference
            # contribute the same way.
            frame_timings = getattr(pipeline, "last_timings", None) or {}
            if frame_timings:
                for key in stage_keys:
                    stage_totals[key] += float(frame_timings.get(key, 0.0))
                stage_frames += 1
            records.append({"frame": frame_index, "predictions": detections})
            if writer is not None or args.show:
                if args.trail_length > 0:
                    # Updated before drawing so the current centre is the newest point.
                    _update_trails(trails, detections, args.trail_length)
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
                    trails=trails if args.trail_length > 0 else None,
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
    if stage_frames:
        averaged = {key: stage_totals[key] / stage_frames for key in stage_keys}
        total_seconds = sum(averaged.values())
        print(
            "Stage averages: "
            f"preprocess {averaged['preprocess'] * 1000:.3f} ms | "
            f"inference {averaged['inference'] * 1000:.3f} ms | "
            f"postprocess {averaged['postprocess'] * 1000:.3f} ms | "
            f"total {total_seconds * 1000:.3f} ms "
            f"({1.0 / total_seconds:.1f} fps)" if total_seconds > 0 else "Stage averages: n/a"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
