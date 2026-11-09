"""Turn a predict run into the payload a DeepLabCut benchmark submission needs.

The DeepLabCut multi-animal benchmark (Lauer et al., Nature Methods 2022) scores a
method by its predictions on the released test images; the ground truth stays
encrypted on their side.  A submission is a pull request against
https://github.com/DeepLabCut/benchmark whose loader returns, per image,

    {image_name: ({"pose": {keypoint_name: (x, y), ...}, "score": 0.917}, ...)}

This script writes the data half of that: one JSON object keyed by image name, with a
list of instances per image.  The loader class itself belongs in the benchmark repo,
not here.

    python tools/dlc_benchmark.py --predictions results/predict/predictions.json \
        --dataset-yaml configs/dataset.yaml --output trimouse-predictions.json

Image names default to ``<folder>_<file>`` because DeepLabCut stores its test frames as
``labeled-data/<video>/<frame>.png`` but indexes them flat as ``<video>_<frame>.png``.
Pass ``--keep-basename`` if your source layout is already flat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _keypoint_names(dataset_yaml: Path, count: int) -> List[str]:
    """Read ``keypoint_names`` from a project dataset file, falling back to indices."""
    names: Sequence[str] = []
    if dataset_yaml is not None:
        try:
            import yaml
        except ModuleNotFoundError as error:  # pragma: no cover - depends on the env
            raise SystemExit(
                "--dataset-yaml needs PyYAML; install it or pass --keypoint-names"
            ) from error
        config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
        names = config.get("keypoint_names") or []
    if names and len(names) == count:
        return [str(name) for name in names]
    if names and len(names) != count:
        raise SystemExit(
            f"keypoint_names lists {len(names)} entries but the predictions carry {count}"
        )
    return [f"kpt_{index}" for index in range(count)]


def _image_key(source: str, keep_basename: bool) -> str:
    path = Path(str(source))
    if keep_basename:
        return path.name
    if path.parent.name:
        return f"{path.parent.name}_{path.name}"
    return path.name


def _prediction_records(payload: Any) -> List[Dict[str, Any]]:
    """Flatten the predict payload into image-level records.

    ``predictions.json`` is a top-level array holding three kinds of entries: image
    results (``source``/``output``/``predictions``), per-video frame results
    (``source``/``frame``/``predictions``) and video summaries (``source``/``output``/
    ``frames``).  Only the first two carry instances.
    """
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise SystemExit("predictions.json must hold an array of records")

    records: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        instances = entry.get("predictions")
        if not instances or "frame" in entry:
            # Video summaries have no per-frame instances; frame records answer to the
            # same source, so the image-level entries are what a benchmark wants.
            continue
        records.append(entry)
    return records


def build_payload(
    predictions_path: Path,
    keypoint_names: List[str],
    keep_basename: bool = False,
    min_score: float = 0.0,
) -> Dict[str, List[Dict[str, Any]]]:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    submission: Dict[str, List[Dict[str, Any]]] = {}
    for record in _prediction_records(payload):
        key = _image_key(record["source"], keep_basename)
        instances: List[Dict[str, Any]] = []
        for instance in record.get("predictions") or []:
            score = float(instance.get("score", 0.0))
            if score < min_score:
                continue
            points = instance.get("keypoints") or []
            if len(points) != len(keypoint_names):
                raise SystemExit(
                    f"{key}: instance has {len(points)} keypoints but the name table has "
                    f"{len(keypoint_names)}"
                )
            pose = {
                name: [round(float(point[0]), 3), round(float(point[1]), 3)]
                for name, point in zip(keypoint_names, points)
            }
            instances.append({"pose": pose, "score": round(score, 6)})
        if instances:
            submission[key] = instances
    return submission


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="predictions.json written by `animalpose-cli predict`")
    parser.add_argument("--output", type=Path, required=True,
                        help="where to write the benchmark payload")
    parser.add_argument("--dataset-yaml", type=Path, default=None,
                        help="project configs/dataset.yaml, read for keypoint_names")
    parser.add_argument("--keypoint-names", default="",
                        help="comma-separated names, used when no dataset YAML is given")
    parser.add_argument("--keep-basename", action="store_true",
                        help="key images by file name only instead of <folder>_<file>")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="drop instances below this confidence")
    args = parser.parse_args(argv)

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    records = _prediction_records(payload)
    if not records:
        raise SystemExit(f"no image-level predictions in {args.predictions}")

    count = len((records[0].get("predictions") or [{}])[0].get("keypoints") or [])
    if args.keypoint_names.strip():
        names = [part.strip() for part in args.keypoint_names.split(",") if part.strip()]
        if len(names) != count:
            raise SystemExit(
                f"--keypoint-names lists {len(names)} entries but the predictions carry {count}"
            )
    else:
        names = _keypoint_names(args.dataset_yaml, count) if count else []

    submission = build_payload(
        args.predictions, names, args.keep_basename, args.min_score
    )
    if not submission:
        raise SystemExit("nothing survived the filters; the payload would be empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Compact on purpose: a benchmark covers thousands of frames, and indenting every
    # coordinate triple quadruples the file for nobody's benefit.
    args.output.write_text(
        json.dumps(submission, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    total = sum(len(instances) for instances in submission.values())
    print(f"{len(submission)} images, {total} instances -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
