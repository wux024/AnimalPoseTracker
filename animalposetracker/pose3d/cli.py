"""``animalpose-cli pose3d``: triangulate the project's 2D predictions into 3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from .calibration import CalibrationError
from .reconstruct import reconstruct_project


def _parse_views(values: Sequence[str]) -> dict:
    views = {}
    for item in values:
        camera, separator, path = str(item).partition("=")
        if not separator or not camera.strip() or not path.strip():
            raise SystemExit(f"--views entries must look like camera=path, got '{item}'")
        views[camera.strip()] = Path(path.strip()).expanduser()
    return views


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="animalpose-cli pose3d",
        description=(
            "Triangulate one predictions.json per camera into results/pose3d of a project "
            "created with --pose3d."
        ),
        epilog=(
            "By default each camera reads <project>/results/predict/<camera>/predictions.json "
            "and the calibration comes from <project>/configs/cameras.yaml."
        ),
    )
    parser.add_argument("--config", default="configs/runtime.yaml", help="project runtime config")
    parser.add_argument("--calibration", help="camera calibration file (default: configs/cameras.yaml)")
    parser.add_argument("--views", action="append", default=[],
                        help="camera=path to that camera's predictions.json (repeatable; point "
                             "it at predictions-filter.json to triangulate the filtered 2D)")
    parser.add_argument("--association", default="index", choices=["index"],
                        help="how instances correspond across cameras (index only for now)")
    parser.add_argument("--output", help="output directory (default: <project>/results/pose3d)")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--reprojection-threshold", type=float, default=8.0)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--pose-filter", default="none", choices=["none", "median", "savgol"],
                        help="temporal filter for the 3D sequence (writes poses3d-filter.json)")
    parser.add_argument("--filter-window", type=int, default=5)
    parser.add_argument("--filter-polyorder", type=int, default=2)
    parser.add_argument("--filter-confidence", type=float)
    parser.add_argument("--filter-max-gap", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = Path(args.config).expanduser().resolve().parent.parent
    pose_filter = None
    if args.pose_filter != "none":
        pose_filter = {
            "method": args.pose_filter,
            "window_length": args.filter_window,
            "polyorder": args.filter_polyorder,
            "confidence_threshold": args.filter_confidence,
            "max_gap": args.filter_max_gap,
        }

    try:
        report = reconstruct_project(
            project_dir,
            calibration_path=args.calibration,
            views=_parse_views(args.views) or None,
            association=args.association,
            output_dir=args.output,
            confidence_threshold=args.confidence_threshold,
            reprojection_threshold=args.reprojection_threshold,
            min_views=args.min_views,
            pose_filter=pose_filter,
        )
    except (CalibrationError, FileNotFoundError, ValueError) as exc:
        print(f"pose3d failed: {exc}")
        return 1

    summary = report["summary"]
    print(
        f"pose3d: {summary['frames']} frames | {summary['instances']} instances | "
        f"{summary['points']} points ({summary['dropped_ratio'] * 100:.1f}% dropped) | "
        f"reprojection median {summary['reprojection_median_px']} px, p90 {summary['reprojection_p90_px']} px"
    )
    print(f"wrote {Path(args.output or project_dir / 'results' / 'pose3d') / 'poses3d.json'}"
          + (" (+ poses3d-filter.json)" if report["outputs"]["poses3d_filtered"] else ""))
    return 0


__all__ = ["run"]
