#!/usr/bin/env python3
"""CLEAR-MOT and IDF1 for ChimpACT, scored video-by-video.

ChimpACT ships one label file per clip (``labels/<clip>.json``) and the matching
continuous video (``videos_full/<clip>.mp4``). A label file covers every tenth frame
(100 frames per 1000-frame video), so:

* the tracker still sees **every** frame, otherwise its motion model is blind between
  annotations and the association numbers are meaningless;
* only the annotated frames are scored -- the other 900 frames have no ground truth and
  counting them would report 900 frames of false positives per clip.

Protocol: every clip is evaluated, including clips whose frames were part of training,
so the numbers are optimistic. ``tracking_metrics.json`` records that.

Usage:
    python tools/chimpact_tracking_eval.py \
        --weights results/onnx/l/best.onnx \
        --config projects/<project>/configs/dataset.yaml \
        --labels /root/wux024/datasets/ChimpACT_release_v1/labels \
        --videos /root/wux024/datasets/ChimpACT_release_v1/videos_full \
        --tracker bytetrack --output results/tracking_eval/bytetrack
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from animalposetracker.inference.session import build_inference_session
from animalposetracker.inference.stream_inference import StreamInferencePipeline
from animalposetracker.tracking.geometry import bbox_iou, linear_assignment

PROTOCOL = "loose-all-clips"
PROTOCOL_NOTE = (
    "Every clip is evaluated, including clips whose frames took part in training. "
    "ChimpACT annotates every 10th frame, so only those frames are scored; the tracker "
    "still runs on every frame."
)
ANNOTATION_STRIDE = 10
MT_THRESHOLD = 0.8
ML_THRESHOLD = 0.2


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def load_clip_labels(label_path: Path) -> Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]:
    """Return {frame_index_in_video: [(gt_id, xyxy), ...]} for one clip.

    ``labels`` indexes frames 0..99; the video index is that number times the stride.
    """
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    image_ids = {image["id"]: index for index, image in enumerate(payload["images"])}
    frames: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = defaultdict(list)
    missing_uuid = 0
    for annotation in payload["annotations"]:
        if int(annotation.get("iscrowd", 0) or 0):
            continue
        index = image_ids.get(annotation["image_id"])
        if index is None:
            continue
        gt_id = annotation.get("bbox_id_ori")
        if not gt_id:
            gt_id = f"ann-{annotation.get('id')}"
            missing_uuid += 1
        x, y, width, height = (float(value) for value in annotation["bbox"])
        frames[index * ANNOTATION_STRIDE].append((str(gt_id), (x, y, x + width, y + height)))
    if missing_uuid:
        print(f"  note: {label_path.name} has {missing_uuid} annotations without bbox_id_ori")
    return dict(frames)


def idf1_metrics(pair_counts, gt_total: int, pred_total: int) -> Tuple[float, float, float, int]:
    from scipy.optimize import linear_sum_assignment

    if gt_total <= 0 or not pair_counts:
        return 0.0, 0.0, 0.0, 0
    gt_ids = sorted({key[0] for key in pair_counts}, key=str)
    pred_ids = sorted({key[1] for key in pair_counts})
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    pred_index = {value: index for index, value in enumerate(pred_ids)}
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for (gt_id, pred_id), count in pair_counts.items():
        counts[gt_index[gt_id], pred_index[pred_id]] = count
    rows, cols = linear_sum_assignment(-counts)
    idtp = int(sum(int(counts[row, col]) for row, col in zip(rows, cols) if counts[row, col] > 0))
    idfp = max(int(pred_total) - idtp, 0)
    idfn = max(int(gt_total) - idtp, 0)
    denominator = 2 * idtp + idfp + idfn
    return (
        2.0 * idtp / denominator if denominator > 0 else 0.0,
        idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0,
        idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0,
        idtp,
    )


def evaluate_clip(
    clip: str,
    label_frames: Dict[int, List[Tuple[str, tuple]]],
    video_path: Path,
    pipeline: StreamInferencePipeline,
    iou_threshold: float,
    max_frames: Optional[int] = None,
) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total_frames = min(total_frames, max_frames)

    gt_total = pred_total = matches_total = 0
    false_positives = false_negatives = id_switches = 0
    iou_sum = 0.0
    pair_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    tracks: Dict[str, dict] = {}
    previous_match: Dict[str, int] = {}
    scored_frames = 0
    frame_index = 0

    while frame_index < total_frames:
        ok, frame = capture.read()
        if not ok:
            break
        # Every frame feeds the tracker; only annotated frames are scored.
        tracked = pipeline.track_frame(frame, frame_index)
        gt_items = label_frames.get(frame_index)
        if gt_items:
            for gt_id, _box in gt_items:
                tracks.setdefault(str(gt_id), {"lifespan": 0, "matched": 0})["lifespan"] += 1
            gt_total += len(gt_items)

            pred_items = [
                (int(item["track_id"]), tuple(float(v) for v in item["bbox_xyxy"]))
                for item in tracked
                if item.get("track_id") is not None and item.get("bbox_xyxy") is not None
            ]
            pred_total += len(pred_items)
            scored_frames += 1

            if pred_items:
                cost = np.ones((len(gt_items), len(pred_items)), dtype=np.float64)
                for row, (_gt_id, gt_box) in enumerate(gt_items):
                    for col, (_pred_id, pred_box) in enumerate(pred_items):
                        cost[row, col] = 1.0 - bbox_iou(gt_box, pred_box)
                matches, unmatched_gt, unmatched_pred = linear_assignment(
                    cost, max_cost=1.0 - float(iou_threshold)
                )
            else:
                matches, unmatched_gt, unmatched_pred = [], list(range(len(gt_items))), []

            false_negatives += len(unmatched_gt)
            false_positives += len(unmatched_pred)
            matches_total += len(matches)

            for row, col in matches:
                gt_id = str(gt_items[row][0])
                pred_id = int(pred_items[col][0])
                iou_sum += 1.0 - float(cost[row, col])
                tracks[gt_id]["matched"] += 1
                pair_counts[(gt_id, pred_id)] += 1
                previous = previous_match.get(gt_id)
                if previous is not None and previous != pred_id:
                    id_switches += 1
                previous_match[gt_id] = pred_id
        frame_index += 1

    capture.release()

    mota = 1.0 - (false_negatives + false_positives + id_switches) / max(gt_total, 1)
    motp = iou_sum / matches_total if matches_total else 0.0
    idf1, idp, idr, idtp = idf1_metrics(pair_counts, gt_total, pred_total)
    track_count = len(tracks)
    mostly_tracked = sum(
        1 for entry in tracks.values() if entry["lifespan"] and entry["matched"] / entry["lifespan"] >= MT_THRESHOLD
    )
    mostly_lost = sum(
        1 for entry in tracks.values() if entry["lifespan"] and entry["matched"] / entry["lifespan"] <= ML_THRESHOLD
    )

    return {
        "clip": clip,
        "frames": frame_index,
        "scored_frames": scored_frames,
        "gt": gt_total,
        "pred": pred_total,
        "matches": matches_total,
        "fp": false_positives,
        "fn": false_negatives,
        "idsw": id_switches,
        "mota": mota,
        "motp_iou": motp,
        "motp_distance": 1.0 - motp,
        "idf1": idf1,
        "idp": idp,
        "idr": idr,
        "idtp": idtp,
        "tracks": track_count,
        "mt": mostly_tracked,
        "ml": mostly_lost,
        "mt_ratio": mostly_tracked / track_count if track_count else 0.0,
        "ml_ratio": mostly_lost / track_count if track_count else 0.0,
    }


def summarize(clip_results: Dict[str, dict]) -> dict:
    totals = {key: 0 for key in ("frames", "scored_frames", "gt", "pred", "matches", "fp", "fn", "idsw")}
    iou_weighted = 0.0
    idtp = tracks = mt = ml = 0
    for result in clip_results.values():
        for key in totals:
            totals[key] += int(result[key])
        iou_weighted += result["motp_iou"] * int(result["matches"])
        idtp += int(result["idtp"])
        tracks += int(result["tracks"])
        mt += int(result["mt"])
        ml += int(result["ml"])

    mota = 1.0 - (totals["fn"] + totals["fp"] + totals["idsw"]) / max(totals["gt"], 1)
    motp = iou_weighted / totals["matches"] if totals["matches"] else 0.0
    idfp = max(totals["pred"] - idtp, 0)
    idfn = max(totals["gt"] - idtp, 0)
    denominator = 2 * idtp + idfp + idfn
    summary = dict(totals)
    summary.update(
        {
            "mota": mota,
            "motp_iou": motp,
            "motp_distance": 1.0 - motp,
            "idtp": idtp,
            "idf1": 2.0 * idtp / denominator if denominator > 0 else 0.0,
            "idp": idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0,
            "idr": idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0,
            "tracks": tracks,
            "mt": mt,
            "ml": ml,
            "mt_ratio": mt / tracks if tracks else 0.0,
            "ml_ratio": ml / tracks if tracks else 0.0,
        }
    )
    return summary


def _self_check() -> int:
    """Synthetic 5-frame sequence: perfect association scores 1, a swap is counted."""
    from types import SimpleNamespace

    class _Echo:
        def __init__(self, mapping_by_frame):
            self.mapping_by_frame = mapping_by_frame

        def track_frame(self, _frame, frame_index):
            mapping = self.mapping_by_frame(frame_index)
            return [
                {"track_id": mapping[index], "bbox_xyxy": list(box)}
                for index, box in enumerate([(0, 0, 10, 10), (20, 20, 30, 30)])
            ]

    label_frames = {
        index * ANNOTATION_STRIDE: [("a", (0, 0, 10, 10)), ("b", (20, 20, 30, 30))]
        for index in range(5)
    }
    with tempfile_video(frames=50) as video:
        perfect = evaluate_clip("x", label_frames, video, _Echo(lambda _f: {0: 1, 1: 2}), 0.5)
        assert abs(perfect["mota"] - 1.0) < 1e-9, perfect
        assert abs(perfect["idf1"] - 1.0) < 1e-9, perfect
        assert perfect["idsw"] == 0, perfect

        swapped = evaluate_clip(
            "x", label_frames, video, _Echo(lambda frame: {0: 1, 1: 2} if frame < 20 else {0: 2, 1: 1}), 0.5
        )
        assert swapped["idsw"] >= 2, swapped
        assert swapped["idf1"] < 1.0, swapped

    print("self-check OK: perfect MOTA/IDF1=1, swap counted")
    return 0


class tempfile_video:
    """Write a tiny black mp4 so the self-check can exercise the real video path."""

    def __init__(self, frames: int = 50, size: Tuple[int, int] = (64, 64)):
        self.frames = frames
        self.size = size
        self.path = None

    def __enter__(self):
        import cv2
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.path = Path(handle.name)
        handle.close()
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, self.size
        )
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        for _ in range(self.frames):
            writer.write(frame)
        writer.release()
        return self.path

    def __exit__(self, *exc_info):
        if self.path and self.path.exists():
            self.path.unlink()
        return False


def _print_table(clip_results: Dict[str, dict], summary: dict) -> None:
    header = (
        f"{'clip':<44}{'frames':>8}{'scored':>7}{'gt':>6}{'pred':>6}"
        f"{'match':>6}{'fp':>5}{'fn':>5}{'idsw':>5}{'MOTA':>8}{'IDF1':>8}"
    )
    print(header)
    print("-" * len(header))
    for clip, result in clip_results.items():
        print(
            f"{clip[:44]:<44}{result['frames']:>8}{result['scored_frames']:>7}"
            f"{result['gt']:>6}{result['pred']:>6}{result['matches']:>6}"
            f"{result['fp']:>5}{result['fn']:>5}{result['idsw']:>5}"
            f"{result['mota']:>8.4f}{result['idf1']:>8.4f}"
        )
    print("-" * len(header))
    print(
        f"{'ALL':<44}{summary['frames']:>8}{summary['scored_frames']:>7}"
        f"{summary['gt']:>6}{summary['pred']:>6}{summary['matches']:>6}"
        f"{summary['fp']:>5}{summary['fn']:>5}{summary['idsw']:>5}"
        f"{summary['mota']:>8.4f}{summary['idf1']:>8.4f}"
    )


def _write_csv(path: Path, clip_results: Dict[str, dict], summary: dict) -> None:
    fields = [
        "clip", "frames", "scored_frames", "gt", "pred", "matches", "fp", "fn", "idsw",
        "mota", "motp_iou", "motp_distance", "idf1", "idp", "idr", "tracks", "mt", "ml",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip, result in clip_results.items():
            row = {"clip": clip}
            row.update({key: result[key] for key in fields if key != "clip"})
            writer.writerow(row)
        total = {"clip": "ALL"}
        total.update({key: summary[key] for key in fields if key != "clip"})
        writer.writerow(total)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", help="exported ONNX artifact of the pose model")
    parser.add_argument("--config", help="dataset.yaml (names / kpt_shape / skeleton)")
    parser.add_argument("--labels", default="/root/wux024/datasets/ChimpACT_release_v1/labels")
    parser.add_argument("--videos", default="/root/wux024/datasets/ChimpACT_release_v1/videos_full")
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="NVIDIA GPU")
    parser.add_argument("--output", help="directory for tracking_metrics.json / tracking_summary.csv")
    parser.add_argument("--max-clips", type=int, help="evaluate only the first N clips")
    parser.add_argument("--max-frames", type=int, help="stop each clip after N frames")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_check:
        return _self_check()
    if not args.weights or not args.config or not args.output:
        parser.error("--weights, --config and --output are required unless --self-check is used")

    labels_dir = Path(args.labels).expanduser()
    videos_dir = Path(args.videos).expanduser()
    label_paths = sorted(labels_dir.glob("*.json"), key=lambda path: natural_key(path.stem))
    if args.max_clips:
        label_paths = label_paths[: args.max_clips]
    if not label_paths:
        raise FileNotFoundError(f"no label json under {labels_dir}")

    session = build_inference_session(
        weights_path=Path(args.weights).expanduser(),
        config_path=Path(args.config).expanduser(),
        engine="onnx",
        device=args.device,
    )
    pipeline = StreamInferencePipeline(
        session.engine,
        session.output_config,
        input_config=session.input_config,
        tracker=args.tracker,
        class_names=list(session.class_names),
    )

    clip_results: Dict[str, dict] = {}
    skipped = []
    for label_path in label_paths:
        clip = label_path.stem
        video_path = videos_dir / f"{clip}.mp4"
        if not video_path.is_file():
            skipped.append(clip)
            continue
        pipeline.reset_tracker()
        clip_results[clip] = evaluate_clip(
            clip,
            load_clip_labels(label_path),
            video_path,
            pipeline,
            args.iou,
            args.max_frames,
        )
        print(
            f"{clip[:44]:<44} MOTA={clip_results[clip]['mota']:.4f} "
            f"IDF1={clip_results[clip]['idf1']:.4f}"
        )

    summary = summarize(clip_results)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "protocol_note": PROTOCOL_NOTE,
        "annotation_stride": ANNOTATION_STRIDE,
        "tracker": args.tracker,
        "iou_threshold": float(args.iou),
        "weights": str(Path(args.weights).expanduser()),
        "device": args.device,
        "skipped_missing_video": skipped,
        "clips": clip_results,
        "overall": summary,
    }
    (output_dir / "tracking_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "tracking_summary.csv", clip_results, summary)
    print()
    _print_table(clip_results, summary)
    print(f"\nprotocol: {PROTOCOL} (optimistic: the detector saw these frames)")
    print(f"written: {output_dir / 'tracking_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
