import subprocess
import sys
import unittest
from types import MappingProxyType

import numpy as np

from animalposetracker.tracking import (
    ALGORITHMS,
    BoTSORT,
    ByteTrack,
    DeepOCSORT,
    FastTracker,
    OCSORT,
    PoseDetection,
    TrackTrack,
    TrackerConfig,
    create_tracker,
)
from animalposetracker.tracking.geometry import linear_assignment, pose_distance, pose_similarity


FRAME = np.zeros((64, 64, 3), dtype=np.uint8)
EXPECTED_TYPES = {
    "bytetrack": ByteTrack,
    "botsort": BoTSORT,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
    "fasttrack": FastTracker,
    "tracktrack": TrackTrack,
}


def detection(
    x=10,
    score=0.9,
    class_id=0,
    keypoints=None,
    embedding=None,
    size=20,
    include_bbox=True,
    include_keypoints=True,
):
    if keypoints is None and include_keypoints:
        keypoints = np.asarray([
            [x + size * 0.2, 15, 0.95],
            [x + size * 0.5, 20, 0.95],
            [x + size * 0.8, 25, 0.95],
        ], dtype=np.float32)
    return PoseDetection(
        bbox_xyxy=(x, 10, x + size, 10 + size) if include_bbox else None,
        score=score,
        class_id=class_id,
        keypoints=keypoints,
        embedding=embedding,
    )


def tracker_config(name, **overrides):
    # GMC is exercised separately; turning it off keeps algorithm tests focused.
    return TrackerConfig.for_algorithm(name, gmc_method="none", **overrides)


class IndependentTrackingTests(unittest.TestCase):
    def test_factory_builds_six_independent_tracker_types_and_defaults_to_bytetrack(self):
        self.assertEqual(set(ALGORITHMS), set(EXPECTED_TYPES))
        self.assertIsInstance(create_tracker(), ByteTrack)
        for name, tracker_type in EXPECTED_TYPES.items():
            with self.subTest(algorithm=name):
                self.assertIsInstance(create_tracker(name), tracker_type)
        check = (
            "import sys; import animalposetracker.tracking; "
            "assert 'ultralytics' not in sys.modules; "
            "assert 'animalposetracker.gui' not in sys.modules; "
            "assert 'animalposetracker.project' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", check], check=True)

    def test_all_default_trackers_run_without_importing_opencv_or_frames(self):
        check = (
            "import builtins; original=builtins.__import__; "
            "builtins.__import__=lambda name,*args,**kwargs: "
            "(_ for _ in ()).throw(AssertionError('cv2 imported')) if name == 'cv2' "
            "else original(name,*args,**kwargs); "
            "from animalposetracker.tracking import ALGORITHMS, create_tracker; "
            "[create_tracker(name).update([{'bbox_xyxy':[1,1,21,21],'score':0.9}], 0) "
            "for name in ALGORITHMS]"
        )
        subprocess.run([sys.executable, "-c", check], check=True)

    def test_public_api_accepts_mapping_observations_and_preserves_detection_fields(self):
        tracker = create_tracker()
        observation = MappingProxyType({
            "bbox_xyxy": (2, 3, 22, 23),
            "score": 0.9,
            "class_id": 4,
            "keypoints": np.asarray([[5, 6, 0.8]], dtype=np.float32),
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        })
        result = tracker.update([observation], 0)
        self.assertEqual(result[0].class_id, 4)
        self.assertEqual(result[0].bbox_xyxy, (2.0, 3.0, 22.0, 23.0))
        mapped = result[0].to_mapping()
        self.assertEqual(mapped["track_id"], 1)
        np.testing.assert_array_equal(mapped["keypoints"], observation["keypoints"])
        np.testing.assert_array_equal(mapped["embedding"], [1.0, 0.0])

    def test_all_algorithms_track_detections_and_keep_ids_across_frames(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                tracker = create_tracker(tracker_config(name))
                frames = 3 if name == "tracktrack" else 2
                results = []
                for frame_index in range(frames):
                    results = tracker.update([detection(x=10 + frame_index)], frame_index)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].track_id, 1)
                self.assertEqual(results[0].class_id, 0)

    def test_all_algorithms_track_pose_detections_without_input_boxes(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                tracker = create_tracker(tracker_config(name))
                results = []
                frames = 3 if name == "tracktrack" else 2
                for frame_index in range(frames):
                    results = tracker.update([
                        detection(x=10 + frame_index, include_bbox=False)
                    ], frame_index)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].track_id, 1)
                self.assertIsNone(results[0].bbox_xyxy)

    def test_all_algorithms_recover_after_a_short_empty_interval(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                tracker = create_tracker(tracker_config(name, track_buffer=4))
                warmup = 3 if name == "tracktrack" else 1
                for frame_index in range(warmup):
                    tracker.update([detection(include_bbox=False)], frame_index)
                tracker.update([], warmup)
                recovered = tracker.update([detection(x=11, include_bbox=False)], warmup + 1)
                self.assertEqual(len(recovered), 1)
                self.assertEqual(recovered[0].track_id, 1)

    def test_low_score_second_association_for_supported_algorithms(self):
        for name in ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack"):
            with self.subTest(algorithm=name):
                options = {"track_low_thresh": 0.10}
                if name in {"ocsort", "deepocsort"}:
                    options["use_byte"] = True
                tracker = create_tracker(tracker_config(name, **options))
                warmup = 3 if name == "tracktrack" else 1
                for frame_index in range(warmup):
                    tracker.update([detection()], frame_index)
                result = tracker.update([detection(x=11, score=0.15)], warmup)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].track_id, 1)

    def test_tracks_are_class_isolated_and_expire_after_buffer(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                tracker = create_tracker(tracker_config(name, track_buffer=1))
                initial = []
                for frame_index in range(3 if name == "tracktrack" else 1):
                    initial = tracker.update(
                        [detection(class_id=0), detection(class_id=1)], frame_index
                    )
                self.assertEqual({item.track_id for item in initial}, {1, 2})
                empty_start = 3 if name == "tracktrack" else 1
                empty_count = 3 if name == "fasttrack" else 2
                for frame_index in range(empty_start, empty_start + empty_count):
                    tracker.update([], frame_index)
                new_start = empty_start + empty_count
                result = []
                for frame_index in range(new_start, new_start + (3 if name == "tracktrack" else 1)):
                    result = tracker.update([detection(class_id=0)], frame_index)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].track_id, 3)

    def test_reset_restarts_sequence_local_ids(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                tracker = create_tracker(tracker_config(name))
                for frame_index in range(3 if name == "tracktrack" else 1):
                    result = tracker.update([detection()], frame_index)
                self.assertEqual(result[0].track_id, 1)
                tracker.reset()
                for frame_index in range(3 if name == "tracktrack" else 1):
                    result = tracker.update([detection(x=30)], frame_index)
                self.assertEqual(result[0].track_id, 1)

    def test_fasttracker_holds_a_track_during_a_short_occlusion(self):
        tracker = create_tracker(tracker_config(
            "fasttrack",
            active_occ_to_lost_thresh=3,
            occ_cover_thresh=0.65,
        ))
        tracker.update([detection(), detection(x=0, size=40)], 0)
        tracker.update([detection(x=0, size=40, score=0.3)], 1)
        hidden = tracker.tracks[1]
        self.assertTrue(hidden.is_occluded)
        self.assertEqual(hidden.state, "tracked")
        tracker.update([], 2)
        recovered = tracker.update([detection(x=11)], 3)
        self.assertTrue(any(item.track_id == 1 for item in recovered))

    def test_tracktrack_tai_suppresses_duplicate_starts(self):
        tracker = create_tracker(tracker_config("tracktrack"))
        tracker.update([detection(), detection(x=11, score=0.8)], 0)
        self.assertEqual(set(tracker.tracks), {1})

    def test_tracktrack_relaxed_lost_rebind_is_opt_in(self):
        for lost_match_thr, expected_state in ((0.0, "lost"), (0.95, "tracked")):
            with self.subTest(lost_match_thr=lost_match_thr):
                tracker = create_tracker(tracker_config(
                    "tracktrack",
                    lost_match_thr=lost_match_thr,
                    kpt_oks_sigmas=(0.1, 0.1, 0.1),
                ))
                for frame_index in range(3):
                    tracker.update([detection()], frame_index)
                tracker.update([], 3)
                tracker.update([detection(x=18)], 4)
                self.assertEqual(tracker.tracks[1].state, expected_state)

    def test_pose_is_primary_and_box_is_an_optional_secondary_cue(self):
        left = np.asarray([[11, 11, 1], [12, 12, 1], [13, 13, 1]], dtype=np.float32)
        right = np.asarray([[11, 11, 1], [12, 12, 0], [13, 13, 1]], dtype=np.float32)
        self.assertAlmostEqual(pose_distance(
            detection(keypoints=left),
            detection(keypoints=right),
            tracker_config("bytetrack", pose_min_common_keypoints=2),
        ), 0.0)
        self.assertTrue(np.isnan(pose_distance(
            detection(keypoints=left),
            detection(keypoints=right),
            tracker_config("bytetrack", pose_min_common_keypoints=3),
        )))

        points_a = np.asarray([[12, 12, 1], [14, 14, 1], [16, 16, 1]], dtype=np.float32)
        points_b = points_a + np.asarray([20, 0, 0], dtype=np.float32)
        for box_weight, expected_ids in ((0.0, [2, 1]), (1.0, [1, 2])):
            tracker = create_tracker(tracker_config(
                "bytetrack", pose_weight=1.0 - box_weight, box_weight=box_weight
            ))
            tracker.update([
                detection(keypoints=points_a),
                detection(keypoints=points_b),
            ], 0)
            result = tracker.update([
                detection(keypoints=points_b),
                detection(keypoints=points_a),
            ], 1)
            self.assertEqual([item.track_id for item in result], expected_ids)

        tight = tracker_config("bytetrack", kpt_oks_sigmas=(0.02, 0.02, 0.02))
        broad = tracker_config("bytetrack", kpt_oks_sigmas=(0.20, 0.20, 0.20))
        points = np.asarray([[10, 10, 1], [15, 15, 1], [20, 20, 1]], dtype=np.float32)
        shifted = points + np.asarray([2, 0, 0], dtype=np.float32)
        self.assertGreater(
            pose_similarity(detection(keypoints=points), detection(keypoints=points), tight),
            pose_similarity(detection(keypoints=points), detection(keypoints=shifted), tight),
        )
        self.assertGreater(
            pose_similarity(detection(keypoints=points), detection(keypoints=shifted), broad),
            pose_similarity(detection(keypoints=points), detection(keypoints=shifted), tight),
        )

    def test_reid_and_gmc_configuration_errors_are_explicit(self):
        reid = create_tracker(tracker_config("botsort", with_reid=True))
        with self.assertRaisesRegex(ValueError, "requires an embedding"):
            reid.update([detection()], 0)

        gmc = create_tracker(TrackerConfig.for_algorithm(
            "botsort", gmc_method="sparseOptFlow"
        ))
        with self.assertRaisesRegex(ValueError, "requires frame"):
            gmc.update([], 0)

    def test_gmc_trackers_accept_frames_and_preserve_ids(self):
        for name in ("botsort", "tracktrack"):
            with self.subTest(algorithm=name):
                tracker = create_tracker(TrackerConfig.for_algorithm(
                    name, gmc_method="sparseOptFlow"
                ))
                repeats = 3 if name == "tracktrack" else 2
                result = []
                for frame_index in range(repeats):
                    result = tracker.update([detection(x=10 + frame_index)], frame_index, frame=FRAME)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].track_id, 1)

        tracker = create_tracker(TrackerConfig.for_algorithm(
            "botsort", gmc_method="sparseOptFlow"
        ))
        result = []
        for frame_index in range(2):
            result = tracker.update(
                [detection(x=10 + frame_index, include_bbox=False)],
                frame_index,
                frame=FRAME,
            )
        self.assertEqual(result[0].track_id, 1)
        self.assertIsNone(result[0].bbox_xyxy)

    def test_reid_features_change_association_and_dimension_mismatch_fails(self):
        tracker = create_tracker(tracker_config("botsort", with_reid=True, appearance_thresh=0.7))
        first = [
            detection(embedding=[1, 0]),
            detection(embedding=[0, 1]),
        ]
        tracker.update(first, 0)
        swapped = [
            detection(embedding=[0, 1]),
            detection(embedding=[1, 0]),
        ]
        result = tracker.update(swapped, 1)
        self.assertEqual([item.track_id for item in result], [2, 1])

        mismatch = create_tracker(tracker_config("botsort", with_reid=True, appearance_thresh=0.0))
        mismatch.update([detection(embedding=[1, 0])], 0)
        with self.assertRaisesRegex(ValueError, "same dimension"):
            mismatch.update([detection(embedding=[1, 0, 0])], 1)

    def test_thresholded_assignment_keeps_valid_pairs_when_other_edges_are_forbidden(self):
        matches, unmatched_rows, unmatched_cols = linear_assignment(
            np.asarray([[0.30, 0.32], [0.32, 0.90]], dtype=np.float32), 0.31
        )
        self.assertEqual(matches, [(0, 0)])
        self.assertEqual(unmatched_rows, [1])
        self.assertEqual(unmatched_cols, [1])

    def test_invalid_algorithm_specific_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not define a ReID"):
            TrackerConfig.for_algorithm("bytetrack", with_reid=True)
        with self.assertRaisesRegex(ValueError, "does not define camera"):
            TrackerConfig.for_algorithm("ocsort", gmc_method="orb")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            TrackerConfig.for_algorithm("bytetrack", delta_t=0)
        defaults = TrackerConfig.for_algorithm("tracktrack")
        self.assertEqual(defaults.lost_match_thr, 0.0)
        self.assertEqual(defaults.tai_oks_thr, 0.85)


if __name__ == "__main__":
    unittest.main()
