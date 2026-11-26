#!/usr/bin/env python3
"""CLEAR-MOT and IDF1 evaluation for the AnimalPoseTracker trackers.

Kept as a tool rather than a package module on purpose: tracking numbers depend
on a protocol choice (which clips, whether the detector ever saw them) that
belongs to the experiment, not to the shipped evaluation layer.

Protocol: **loose-all-clips**. Every clip of the split is evaluated, including
clips whose frames were part of training, so the numbers are optimistic by
construction. ``tracking_metrics.json`` records that in ``protocol_note``.

GT identity comes from ``bbox_id_ori`` (a per-chimpanzee UUID) in the COCO
annotations. Predictions come from ``predict --split <split>``'s
``predictions.json``, so the pose model runs once and every tracker is compared
on identical detections.

Usage:
    python tools/tracking_eval.py \
        --predictions results/pose_vis/l --config <project>/configs/dataset.yaml \
        --split test --tracker bytetrack --iou 0.5 --output results/tracking_eval/bytetrack

    python tools/tracking_eval.py --self-check
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

from animalposetracker.tracking.algorithms import create_tracker
from animalposetracker.tracking.geometry import bbox_iou, linear_assignment
from animalposetracker.tracking.types import PoseDetection

PROTOCOL = "loose-all-clips"
PROTOCOL_NOTE = (
    "All clips of the split were evaluated, including clips whose frames took part in "
    "training; the pose model has seen them, so these numbers are optimistic and are not "
    "comparable to a held-out protocol."
)

MT_THRESHOLD = 0.8
ML_THRESHOLD = 0.2


# --------------------------------------------------------------------------- helpers


def natural_key(text: str) -> list:
    """Sort key that treats digit runs as integers ('frame_9' before 'frame_10')."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def read_dataset_yaml(path: Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"dataset yaml must be a mapping: {path}")
    return values


def resolve_dataset_root(config_path: Path, values: dict) -> Path:
    raw = values.get("path")
    if raw is None:
        return Path(config_path).resolve().parent.parent
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(config_path).resolve().parent.parent / candidate
    return candidate.resolve()


def load_ground_truth(config_path: Path, split: str) -> dict:
    """Collect per-clip GT boxes keyed by frame name, identified by ``bbox_id_ori``."""
    config_path = Path(config_path).expanduser().resolve()
    values = read_dataset_yaml(config_path)
    root = resolve_dataset_root(config_path, values)
    annotation_path = root / "annotations" / f"{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"annotation json does not exist: {annotation_path}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data.get("images", [])}
    clips: Dict[str, dict] = {}
    skipped_crowd = 0
    missing_uuid = 0

    for annotation in data.get("annotations", []):
        if int(annotation.get("iscrowd", 0) or 0):
            skipped_crowd += 1
            continue
        image = images.get(annotation.get("image_id"))
        if image is None:
            continue
        parts = [p for p in str(image.get("file_name", "")).replace("\\", "/").split("/") if p]
        if len(parts) < 2:
            continue
        clip, frame_name = parts[-2], parts[-1]
        gt_id = annotation.get("bbox_id_ori")
        if not gt_id:
            gt_id = f"ann-{annotation.get('id')}"
            missing_uuid += 1
        x, y, width, height = (float(value) for value in annotation["bbox"])
        clip_entry = clips.setdefault(clip, {"frames": {}, "gt_ids": set()})
        clip_entry["frames"].setdefault(frame_name, []).append(
            (str(gt_id), (x, y, x + width, y + height))
        )
        clip_entry["gt_ids"].add(str(gt_id))

    for entry in clips.values():
        entry["frame_names"] = sorted(entry["frames"], key=natural_key)
    return {
        "root": root,
        "clips": clips,
        "skipped_crowd": skipped_crowd,
        "missing_uuid": missing_uuid,
    }


def load_prediction_records(paths: Sequence) -> List[dict]:
    entries: List[dict] = []
    for raw in paths:
        path = Path(raw).expanduser()
        candidates = [path] if path.is_file() else sorted(path.rglob("predictions.json"))
        for candidate in candidates:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            for entry in data:
                if isinstance(entry, dict) and "predictions" in entry:
                    entries.append(entry)
    return entries


def index_predictions(entries: Sequence[dict]) -> Tuple[dict, dict]:
    """Index predictions by resolved path and, as a fallback, by (clip, frame)."""
    by_path: Dict[str, list] = {}
    by_clip_frame: Dict[Tuple[str, str], list] = {}
    for entry in entries:
        source = str(entry.get("source") or "")
        if not source:
            continue
        predictions = entry.get("predictions") or []
        path = Path(source)
        try:
            by_path[str(path.resolve())] = predictions
        except OSError:
            pass
        parts = [p for p in path.as_posix().split("/") if p]
        if len(parts) >= 2:
            by_clip_frame[(parts[-2], parts[-1])] = predictions
    return by_path, by_clip_frame


def build_detections(records: Sequence[dict]) -> List[PoseDetection]:
    detections: List[PoseDetection] = []
    for record in records:
        box = record.get("bbox_xyxy")
        if box is None:
            continue
        values = [float(value) for value in box]
        if values[2] <= values[0] or values[3] <= values[1]:
            continue
        score = record.get("score")
        score = 1.0 if score is None else min(max(float(score), 0.0), 1.0)
        detections.append(
            PoseDetection.from_mapping(
                {
                    "bbox_xyxy": values,
                    "score": score,
                    "class_id": int(record.get("class_id") or 0),
                    "keypoints": record.get("keypoints"),
                }
            )
        )
    return detections


# --------------------------------------------------------------------------- core


def idf1_metrics(
    pair_counts: Dict[Tuple[str, int], int], gt_total: int, pred_total: int
) -> Tuple[float, float, float, int]:
    """Global ID matching: maximise matched detections over GT-id x pred-id counts."""
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
    idf1 = 2.0 * idtp / denominator if denominator > 0 else 0.0
    idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
    return idf1, idp, idr, idtp


def evaluate_sequence(
    gt_frames: Dict[str, List[Tuple[str, tuple]]],
    frame_names: Sequence[str],
    prediction_frames: Sequence[Sequence[dict]],
    tracker,
    iou_threshold: float,
) -> dict:
    """Run one clip through a tracker and score it against its frames of GT."""

    tracker.reset()
    gt_total = 0
    pred_total = 0
    matches_total = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    iou_sum = 0.0
    pair_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    tracks: Dict[str, dict] = {}
    previous_match: Dict[str, int] = {}
    processed_frames = 0

    for frame_index, frame_name in enumerate(frame_names):
        gt_items = gt_frames.get(frame_name, [])
        for gt_id, _box in gt_items:
            entry = tracks.setdefault(str(gt_id), {"lifespan": 0, "matched": 0})
            entry["lifespan"] += 1
        gt_total += len(gt_items)

        detections = build_detections(prediction_frames[frame_index])
        tracked = tracker.update(detections, frame_index, frame=None)
        pred_items = [
            (int(item.track_id), tuple(float(v) for v in item.bbox_xyxy))
            for item in tracked
            if item.bbox_xyxy is not None
        ]
        pred_total += len(pred_items)
        processed_frames += 1

        if gt_items and pred_items:
            cost = np.ones((len(gt_items), len(pred_items)), dtype=np.float64)
            for row, (_gt_id, gt_box) in enumerate(gt_items):
                for col, (_pred_id, pred_box) in enumerate(pred_items):
                    cost[row, col] = 1.0 - bbox_iou(gt_box, pred_box)
            matches, unmatched_gt, unmatched_pred = linear_assignment(
                cost, max_cost=1.0 - float(iou_threshold)
            )
        else:
            matches, unmatched_gt, unmatched_pred = (
                [],
                list(range(len(gt_items))),
                list(range(len(pred_items))),
            )

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

    mota = 1.0 - (false_negatives + false_positives + id_switches) / max(gt_total, 1)
    motp_iou = iou_sum / matches_total if matches_total else 0.0
    idf1, idp, idr, idtp = idf1_metrics(pair_counts, gt_total, pred_total)

    mostly_tracked = sum(
        1 for entry in tracks.values() if entry["lifespan"] and entry["matched"] / entry["lifespan"] >= MT_THRESHOLD
    )
    mostly_lost = sum(
        1 for entry in tracks.values() if entry["lifespan"] and entry["matched"] / entry["lifespan"] <= ML_THRESHOLD
    )
    track_count = len(tracks)

    return {
        "frames": processed_frames,
        "gt": gt_total,
        "pred": pred_total,
        "matches": matches_total,
        "fp": false_positives,
        "fn": false_negatives,
        "idsw": id_switches,
        "mota": mota,
        "motp_iou": motp_iou,
        "motp_distance": 1.0 - motp_iou,
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
    """Micro-average over clips: counts add up, ratios and MOTA are recomputed."""
    totals = {
        "frames": 0,
        "gt": 0,
        "pred": 0,
        "matches": 0,
        "fp": 0,
        "fn": 0,
        "idsw": 0,
    }
    iou_weighted = 0.0
    idtp = 0
    tracks = 0
    mt = 0
    ml = 0
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


# --------------------------------------------------------------------------- self-check


def _self_check() -> int:
    """Synthetic clips: a perfect tracker must score 1, an identity swap must be counted."""

    class _Echo:
        """Stand-in tracker returning detections unchanged with a per-frame id mapping."""

        def __init__(self, mapping_by_frame):
            self.mapping_by_frame = mapping_by_frame

        def reset(self):
            pass

        def update(self, detections, frame_index, frame=None):
            del frame
            from types import SimpleNamespace

            mapping = self.mapping_by_frame(frame_index)
            output = []
            for index, detection in enumerate(detections):
                track_id = mapping.get(index, 100 + index)
                output.append(
                    SimpleNamespace(track_id=track_id, bbox_xyxy=list(detection.bbox_xyxy))
                )
            return output

    def frames(count=5):
        gt = {}
        names = []
        predictions = []
        for index in range(count):
            name = f"{index:06d}.jpg"
            names.append(name)
            gt[name] = [
                ("a", (0.0, 0.0, 10.0, 10.0)),
                ("b", (20.0, 20.0, 30.0, 30.0)),
            ]
            predictions.append(
                [
                    {"bbox_xyxy": [0, 0, 10, 10], "score": 0.9, "class_id": 0, "keypoints": None},
                    {"bbox_xyxy": [20, 20, 30, 30], "score": 0.9, "class_id": 0, "keypoints": None},
                ]
            )
        return gt, names, predictions

    # Perfect association.
    gt, names, predictions = frames()
    perfect = evaluate_sequence(gt, names, predictions, _Echo(lambda _f: {0: 1, 1: 2}), 0.5)
    assert abs(perfect["mota"] - 1.0) < 1e-9, perfect
    assert abs(perfect["idf1"] - 1.0) < 1e-9, perfect
    assert perfect["idsw"] == 0, perfect

    # Identity swap from frame 2 onward: same boxes, ids exchanged mid-sequence.
    swapped = evaluate_sequence(
        gt,
        names,
        predictions,
        _Echo(lambda frame: {0: 1, 1: 2} if frame < 2 else {0: 2, 1: 1}),
        0.5,
    )
    assert swapped["idsw"] >= 2, swapped
    assert swapped["idf1"] < 1.0, swapped

    # One prediction dropped per frame: exactly one more false negative per frame.
    one_box = [frame[:1] for frame in predictions]
    missing = evaluate_sequence(gt, names, one_box, _Echo(lambda _f: {0: 1}), 0.5)
    assert missing["fn"] == len(names), missing
    assert missing["fp"] == 0, missing

    print("self-check OK: perfect MOTA/IDF1=1, swap counted, missing box counted")
    return 0


# --------------------------------------------------------------------------- CLI


def _print_table(clip_results: Dict[str, dict], summary: dict) -> None:
    header = (
        f"{'clip':<40} {'frames':>6} {'gt':>6} {'pred':>6} {'match':>6} "
        f"{'fp':>5} {'fn':>5} {'idsw':>5} {'MOTA':>7} {'IDF1':>7}"
    )
    print(header)
    print("-" * len(header))
    for clip, result in clip_results.items():
        print(
            f"{clip[:40]:<40} {result['frames']:>6} {result['gt']:>6} {result['pred']:>6} "
            f"{result['matches']:>6} {result['fp']:>5} {result['fn']:>5} {result['idsw']:>5} "
            f"{result['mota']:>7.4f} {result['idf1']:>7.4f}"
        )
    print("-" * len(header))
    print(
        f"{'ALL':<40} {summary['frames']:>6} {summary['gt']:>6} {summary['pred']:>6} "
        f"{summary['matches']:>6} {summary['fp']:>5} {summary['fn']:>5} {summary['idsw']:>5} "
        f"{summary['mota']:>7.4f} {summary['idf1']:>7.4f}"
    )


def _write_csv(path: Path, clip_results: Dict[str, dict], summary: dict) -> None:
    fields = [
        "clip", "frames", "gt", "pred", "matches", "fp", "fn", "idsw",
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", help="predictions.json file(s) or directories to search")
    parser.add_argument("--config", help="dataset.yaml of the project (locates annotations and images)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--iou", type=float, default=0.5, help="association IoU threshold")
    parser.add_argument("--output", help="directory for tracking_metrics.json / tracking_summary.csv")
    parser.add_argument("--max-frames", type=int, help="stop each clip after N frames")
    parser.add_argument("--max-clips", type=int, help="evaluate only the first N clips")
    parser.add_argument("--self-check", action="store_true", help="run the built-in assertions and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_check:
        return _self_check()
    if not args.predictions or not args.config or not args.output:
        parser.error("--predictions, --config and --output are required unless --self-check is used")

    ground_truth = load_ground_truth(Path(args.config), args.split)
    root = ground_truth["root"]
    entries = load_prediction_records(args.predictions)
    by_path, by_clip_frame = index_predictions(entries)

    clip_names = sorted(ground_truth["clips"], key=natural_key)
    if args.max_clips:
        clip_names = clip_names[: args.max_clips]

    tracker = create_tracker({"algorithm": args.tracker})
    clip_results: Dict[str, dict] = {}
    missing_frames: Dict[str, int] = {}

    for clip in clip_names:
        entry = ground_truth["clips"][clip]
        frame_names = entry["frame_names"]
        if args.max_frames:
            frame_names = frame_names[: args.max_frames]
        prediction_frames = []
        missing = 0
        for frame_name in frame_names:
            expected = (root / "images" / args.split / clip / frame_name).resolve()
            records = by_path.get(str(expected))
            if records is None:
                records = by_clip_frame.get((clip, frame_name), [])
                if not records:
                    missing += 1
            prediction_frames.append(records)
        if missing:
            missing_frames[clip] = missing
        clip_results[clip] = evaluate_sequence(
            entry["frames"], frame_names, prediction_frames, tracker, args.iou
        )

    summary = summarize(clip_results)

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "protocol_note": PROTOCOL_NOTE,
        "tracker": args.tracker,
        "split": args.split,
        "iou_threshold": float(args.iou),
        "dataset_root": str(root),
        "prediction_sources": [str(Path(value).expanduser()) for value in args.predictions],
        "skipped_crowd": ground_truth["skipped_crowd"],
        "gt_without_uuid": ground_truth["missing_uuid"],
        "missing_frames": missing_frames,
        "clips": clip_results,
        "overall": summary,
    }
    (output_dir / "tracking_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "tracking_summary.csv", clip_results, summary)

    _print_table(clip_results, summary)
    print(f"\nprotocol: {PROTOCOL} (optimistic: detector saw these frames)")
    print(f"written: {output_dir / 'tracking_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
