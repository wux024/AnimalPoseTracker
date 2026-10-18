"""Cross-camera instance association: identity is geometry, not detection order."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from animalposetracker.pose3d import (
    CameraCalibration,
    CameraRig,
    associate_instances,
    reconstruct_project,
)

INTRINSIC = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
EYES = {"camA": (1.2, 0.0, 0.3), "camB": (-0.9, 0.8, 0.3), "camC": (-0.5, -1.0, 0.2)}


def _look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(up, float), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward]), -np.stack([right, down, forward]) @ eye


def _rig(eyes=EYES):
    cameras = {name: CameraCalibration(name, INTRINSIC, *_look_at(eye))
               for name, eye in eyes.items()}
    return CameraRig(cameras=cameras,
                     distortion={name: ("none", np.zeros(4)) for name in cameras})


def _project(points, camera):
    """``(K, 3)`` world points to ``(K, 3)`` pixels with confidence 1."""
    local = points @ camera.rotation_matrix.T + camera.translation_vector
    pixels = (local / local[:, 2:3]) @ camera.intrinsic_matrix.T
    return np.column_stack([pixels[:, :2], np.ones(len(points))])


def _keypoints(pixels):
    """``(K, 2)`` pixels as the ``[u, v, confidence]`` rows a predictions file holds."""
    return [[float(x), float(y), 1.0] for x, y in pixels]


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.rig = _rig()
        rng = np.random.default_rng(7)
        self.animal_a = rng.normal(scale=0.08, size=(6, 3))
        self.animal_b = self.animal_a + np.array([0.25, 0.0, 0.0])

    def _poses(self, **instances):
        """Build per-camera instance lists from keyword names."""
        return {camera: [getattr(self, name) for name in names]
                for camera, names in instances.items()}

    def test_one_animal_per_frame_pairs_every_camera(self):
        poses = {name: [_project(self.animal_a, camera)] for name, camera in self.rig.cameras.items()}
        groups = associate_instances(poses, self.rig)
        self.assertEqual(groups, [{"camA": 0, "camB": 0, "camC": 0}])

    def test_detection_order_does_not_decide_identity(self):
        # camB reports the animals in the opposite order; index pairing would swap them.
        poses = {
            "camA": [_project(self.animal_a, self.rig.cameras["camA"]),
                     _project(self.animal_b, self.rig.cameras["camA"])],
            "camB": [_project(self.animal_b, self.rig.cameras["camB"]),
                     _project(self.animal_a, self.rig.cameras["camB"])],
        }
        groups = associate_instances(poses, self.rig)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], {"camA": 0, "camB": 1})
        self.assertEqual(groups[1], {"camA": 1, "camB": 0})

    def test_a_missing_detection_keeps_the_other_animals_paired(self):
        # camC only sees animal A, so its single instance must join A's group.
        poses = {
            "camA": [_project(self.animal_a, self.rig.cameras["camA"]),
                     _project(self.animal_b, self.rig.cameras["camA"])],
            "camB": [_project(self.animal_b, self.rig.cameras["camB"]),
                     _project(self.animal_a, self.rig.cameras["camB"])],
            "camC": [_project(self.animal_a, self.rig.cameras["camC"])],
        }
        groups = associate_instances(poses, self.rig)
        self.assertEqual(groups[0], {"camA": 0, "camB": 1, "camC": 0})
        self.assertEqual(groups[1], {"camA": 1, "camB": 0})

    def test_groups_seen_by_too_few_cameras_are_dropped(self):
        poses = {
            "camA": [_project(self.animal_a, self.rig.cameras["camA"]),
                     _project(self.animal_b, self.rig.cameras["camA"])],
            "camB": [_project(self.animal_b, self.rig.cameras["camB"]),
                     _project(self.animal_a, self.rig.cameras["camB"])],
            "camC": [_project(self.animal_a, self.rig.cameras["camC"])],
        }
        groups = associate_instances(poses, self.rig, min_views=3)
        self.assertEqual(groups, [{"camA": 0, "camB": 1, "camC": 0}])

    def test_a_pairing_worse_than_the_threshold_is_refused(self):
        far = _project(self.animal_b, self.rig.cameras["camB"]).copy()
        far[:, 0] += 40.0
        poses = {
            "camA": [_project(self.animal_a, self.rig.cameras["camA"])],
            "camB": [far],
        }
        self.assertEqual(associate_instances(poses, self.rig), [])
        self.assertEqual(len(associate_instances(poses, self.rig, max_sampson_px=80.0)), 1)

    def test_low_confidence_keypoints_do_not_decide_the_pairing(self):
        noisy = _project(self.animal_b, self.rig.cameras["camB"]).copy()
        noisy[:, :2] += np.array([30.0, -25.0])
        noisy[:, 2] = 0.01        # nearly all of it is untrustworthy
        poses = {
            "camA": [_project(self.animal_a, self.rig.cameras["camA"])],
            "camB": [noisy],
        }
        self.assertEqual(associate_instances(poses, self.rig), [])
        self.assertEqual(len(associate_instances(poses, self.rig, confidence_threshold=0.0,
                                                 max_sampson_px=60.0)), 1)

    def test_three_cameras_agree_on_the_same_animal(self):
        poses = {
            "camA": [_project(self.animal_b, self.rig.cameras["camA"]),
                     _project(self.animal_a, self.rig.cameras["camA"])],
            "camB": [_project(self.animal_a, self.rig.cameras["camB"]),
                     _project(self.animal_b, self.rig.cameras["camB"])],
            "camC": [_project(self.animal_b, self.rig.cameras["camC"]),
                     _project(self.animal_a, self.rig.cameras["camC"])],
        }
        groups = associate_instances(poses, self.rig)
        self.assertEqual(groups, [
            {"camA": 0, "camB": 1, "camC": 0},
            {"camA": 1, "camB": 0, "camC": 1},
        ])

    def test_one_bad_camera_does_not_cost_the_whole_frame(self):
        # A rig whose first camera carries a large residual must still pair the
        # remaining two, instead of dropping the frame entirely.
        broken = _project(self.animal_a, self.rig.cameras["camA"]).copy()
        # jitter across the epipolar lines, not along them: a shift along a line
        # would keep the Sampson distance small and prove nothing
        broken[:, :2] += np.random.default_rng(1).normal(scale=40.0, size=(len(broken), 2))
        poses = {
            "camA": [broken],
            "camB": [_project(self.animal_a, self.rig.cameras["camB"])],
            "camC": [_project(self.animal_a, self.rig.cameras["camC"])],
        }
        self.assertEqual(associate_instances(poses, self.rig, max_sampson_px=10.0),
                         [{"camB": 0, "camC": 0}])
        self.assertEqual(associate_instances(poses, self.rig, max_sampson_px=10.0, min_views=3), [])

    def test_bad_parameters_are_rejected(self):
        poses = {name: [_project(self.animal_a, camera)] for name, camera in self.rig.cameras.items()}
        with self.assertRaises(ValueError):
            associate_instances(poses, self.rig, max_sampson_px=0.0)
        with self.assertRaises(ValueError):
            associate_instances(poses, self.rig, min_points=3)
        with self.assertRaises(ValueError):
            associate_instances(poses, self.rig, confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            associate_instances(poses, self.rig, min_views=1)

    def test_an_empty_frame_associates_to_nothing(self):
        self.assertEqual(associate_instances({}, self.rig), [])
        self.assertEqual(associate_instances({"camA": [], "camB": []}, self.rig), [])


class ReconstructionAssociationTests(unittest.TestCase):
    def _project(self, root: Path, eyes, records) -> Path:
        project = root / "probe-test-AnimalRTPose-N-20260101"
        (project / "configs").mkdir(parents=True)
        (project / "results").mkdir()
        (project / "project.yaml").write_text(
            yaml.safe_dump({"project_name": "probe", "pose3d": True}), encoding="utf-8"
        )
        blocks = {}
        for name, eye in eyes.items():
            rotation, translation = _look_at(eye)
            blocks[name] = {"intrinsic": INTRINSIC.tolist(), "rotation": rotation.tolist(),
                            "translation": translation.tolist(), "distortion": {"model": "none"}}
        (project / "configs" / "cameras.yaml").write_text(
            yaml.safe_dump({"world_unit": "mm", "cameras": blocks}), encoding="utf-8"
        )
        for name, payload in records.items():
            target = project / "results" / "predict" / name
            target.mkdir(parents=True)
            (target / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")
        return project

    def _frames(self, project: Path):
        poses = json.loads((project / "results" / "pose3d" / "poses3d.json").read_text(encoding="utf-8"))
        return [np.asarray(instance["keypoints_3d"], dtype=np.float64)
                for instance in poses["frames"][0]["instances"]]

    def test_epipolar_recovers_two_animals_index_would_swap(self):
        eyes = {"camA": (1.2, 0.0, 0.3), "camB": (-0.9, 0.8, 0.3)}
        rig = _rig(eyes)
        rng = np.random.default_rng(5)
        animal_a = rng.normal(scale=0.08, size=(5, 3))
        animal_b = animal_a + np.array([0.25, 0.0, 0.0])
        a_pixels = _project(animal_a, rig.cameras["camA"])[:, :2]
        b_pixels = _project(animal_b, rig.cameras["camA"])[:, :2]
        # camB lists the animals the other way round, as a real detector may
        swapped_a = _project(animal_b, rig.cameras["camB"])[:, :2]
        swapped_b = _project(animal_a, rig.cameras["camB"])[:, :2]

        records = {
            "camA": [{"source": "0.png", "predictions": [
                {"keypoints": _keypoints(a_pixels)}, {"keypoints": _keypoints(b_pixels)}]}],
            "camB": [{"source": "0.png", "predictions": [
                {"keypoints": _keypoints(swapped_a)}, {"keypoints": _keypoints(swapped_b)}]}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary), eyes, records)
            report = reconstruct_project(project, association="epipolar")
            recovered = self._frames(project)
            self.assertEqual(report["association"], "epipolar")
            self.assertEqual(report["association_stats"]["matched_instances"], 4)
            self.assertEqual(report["association_stats"]["unmatched_instances"], {"camA": 0, "camB": 0})
            self.assertEqual(len(recovered), 2)
            self.assertLess(float(np.abs(recovered[0] - animal_a).max()), 1e-3)
            self.assertLess(float(np.abs(recovered[1] - animal_b).max()), 1e-3)
            matched = json.loads(
                (project / "results" / "pose3d" / "poses3d.json").read_text(encoding="utf-8")
            )["frames"][0]["instances"][0]["view_indices"]
            self.assertEqual(matched, {"camA": 0, "camB": 1})

        # the same frame under index association triangulates the wrong pairs
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary), eyes, records)
            reconstruct_project(project, association="index")
            swapped = self._frames(project)
            self.assertGreater(float(np.abs(swapped[0] - animal_a).max()), 0.02)

    def test_a_single_animal_gives_identical_geometry_under_both_modes(self):
        eyes = {"camA": (1.2, 0.0, 0.3), "camB": (-0.9, 0.8, 0.3)}
        rig = _rig(eyes)
        points = np.random.default_rng(9).normal(scale=0.08, size=(5, 3))
        records = {}
        for name, camera in rig.cameras.items():
            pixels = _project(points, camera)[:, :2]
            records[name] = [{"source": "0.png", "predictions": [
                {"keypoints": _keypoints(pixels)}]}]
        results = {}
        for mode in ("index", "epipolar"):
            with tempfile.TemporaryDirectory() as temporary:
                project = self._project(Path(temporary), eyes, records)
                reconstruct_project(project, association=mode)
                results[mode] = self._frames(project)
        self.assertEqual(np.asarray(results["index"][0]).shape,
                         np.asarray(results["epipolar"][0]).shape)
        self.assertLess(float(np.abs(results["index"][0] - results["epipolar"][0]).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
