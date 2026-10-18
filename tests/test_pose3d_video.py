"""The 3D renderers: the overlay projects through the rig, the 3D view needs no video."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

import cv2

from animalposetracker.pose3d import CameraCalibration, CameraRig
from animalposetracker.pose3d.render import (
    load_skeleton,
    render_overlay,
    render_space,
    source_videos,
)

INTRINSIC = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(up, float), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ eye


def _rig(eyes):
    cameras = {name: CameraCalibration(name, INTRINSIC, *_look_at(eye))
               for name, eye in eyes.items()}
    return CameraRig(cameras=cameras,
                     distortion={name: ("none", np.zeros(4)) for name in cameras})


def _skeleton_points(frames=12):
    """A small stick figure that drifts sideways, in world units the rig can see."""
    rng = np.random.default_rng(4)
    base = rng.normal(scale=0.05, size=(4, 3))
    return [base + np.array([0.02 * step, 0.0, 0.0]) for step in range(frames)]


def _write_video(path: Path, frames: int, size=(640, 480)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened(), "test environment cannot write mp4"
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        cv2.putText(frame, str(index), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        writer.write(frame)
    writer.release()


def _payload(points_by_frame, cameras):
    frames = []
    for index, points in enumerate(points_by_frame):
        frames.append({
            "frame": index,
            "instances": [{
                "keypoints_3d": np.round(points, 4).tolist(),
                "valid": [True] * len(points),
                "views": sorted(cameras),
            }],
        })
    return {"schema": "pose3d/1", "world_unit": "mm", "frames": frames}


class SkeletonTests(unittest.TestCase):
    def test_skeleton_is_read_from_the_dataset_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "configs").mkdir()
            (project / "configs" / "dataset.yaml").write_text(
                yaml.safe_dump({"skeleton": [[0, 1], [1, 2], [2, 3]]}), encoding="utf-8"
            )
            self.assertEqual(load_skeleton(project), [(0, 1), (1, 2), (2, 3)])

    def test_a_missing_or_broken_skeleton_is_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self.assertEqual(load_skeleton(project), [])
            (project / "configs").mkdir()
            (project / "configs" / "dataset.yaml").write_text(
                yaml.safe_dump({"skeleton": "none"}), encoding="utf-8"
            )
            self.assertEqual(load_skeleton(project), [])


class SourceVideoTests(unittest.TestCase):
    def test_the_recorded_source_names_the_video(self):
        records = {"camA": [{"source": "run.mp4", "frame": 0, "predictions": []},
                            {"source": "run.mp4", "output": "out.mp4", "frames": 10}]}
        self.assertEqual(source_videos(records), {"camA": Path("run.mp4")})

    def test_image_sources_are_ignored(self):
        records = {"camA": [{"source": "frame0001.png", "predictions": []}]}
        self.assertEqual(source_videos(records), {})


class RenderTests(unittest.TestCase):
    EYES = {"camA": (1.2, 0.0, 0.3), "camB": (-0.9, 0.8, 0.3)}

    def test_the_3d_view_writes_a_video_without_any_source_video(self):
        rig = _rig(self.EYES)
        payload = _payload(_skeleton_points(12), rig.cameras)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "space.mp4"
            frames = render_space(payload, target, skeleton=[(0, 1), (1, 2), (2, 3)],
                                  fps=10.0, width=320, height=240)
            self.assertEqual(frames, 12)
            self.assertGreater(target.stat().st_size, 0)
            capture = cv2.VideoCapture(str(target))
            self.assertTrue(capture.isOpened())
            self.assertGreater(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            capture.release()

    def test_the_overlay_projects_onto_each_camera_video(self):
        rig = _rig(self.EYES)
        points = _skeleton_points(10)
        payload = _payload(points, rig.cameras)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for camera in rig.cameras:
                path = root / f"{camera}.mp4"
                _write_video(path, len(points))
                sources[camera] = path
            target = root / "overlay.mp4"
            frames = render_overlay(payload, rig, sources, target,
                                    skeleton=[(0, 1), (1, 2), (2, 3)], fps=10.0, max_width=320)
            self.assertEqual(frames, 10)
            self.assertGreater(target.stat().st_size, 0)
            capture = cv2.VideoCapture(str(target))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            # two 320-wide tiles side by side
            self.assertEqual(frame.shape[1], 640)

    def test_striding_keeps_every_nth_frame(self):
        rig = _rig(self.EYES)
        payload = _payload(_skeleton_points(12), rig.cameras)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "space.mp4"
            frames = render_space(payload, target, skeleton=[], fps=10.0,
                                  width=240, height=180, stride=4)
            self.assertEqual(frames, 3)

    def test_an_empty_sequence_is_refused(self):
        rig = _rig(self.EYES)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                render_space({"frames": []}, Path(temporary) / "empty.mp4")

    def test_the_overlay_needs_at_least_one_source(self):
        rig = _rig(self.EYES)
        payload = _payload(_skeleton_points(4), rig.cameras)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                render_overlay(payload, rig, {}, Path(temporary) / "none.mp4")

    def test_stride_must_be_positive(self):
        rig = _rig(self.EYES)
        payload = _payload(_skeleton_points(4), rig.cameras)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                render_space(payload, Path(temporary) / "x.mp4", stride=0)


if __name__ == "__main__":
    unittest.main()
