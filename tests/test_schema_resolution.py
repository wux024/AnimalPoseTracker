"""Tests for the inferencer's pose-output-schema resolution.

The resolver decides which of the three ultralytics layouts an artifact emits.
That decision drives the channel arithmetic in the decoder, so every branch is
pinned here: the precedence chain, the three shape outcomes, and the ``nc == 2``
ambiguity that makes shape-only detection dangerous.
"""

import unittest
import warnings

from animalposetracker.postprocessing.output_decoders import (
    RTMO_MMDEPLOY,
    RTMPOSE_SIMCC,
    YOLO_POSE_END2END,
    YOLO_POSE_NMS,
    YOLO_POSE_RAW,
)
from animalposetracker.inference.schema_resolution import (
    RESERVED_SCHEMAS,
    SUPPORTED_YOLO_SCHEMAS,
    resolve_output_schema,
)


# A realistic animal pose head: 17 keypoints with x, y, visibility.
KPT, DIMS = 17, 3
RAW_BASE = 4 + KPT * DIMS        # 55 channels before the class slots
PROCESSED = 6 + KPT * DIMS       # 57 channels after NMS / top-k
ANCHORS = 8400
MAX_DET = 300


def resolve(**kwargs):
    kwargs.setdefault("num_keypoints", KPT)
    kwargs.setdefault("keypoint_dims", DIMS)
    return resolve_output_schema(**kwargs)


def caught_warnings(call):
    """Run ``call`` and return ``(result, [warning messages])``."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = call()
    return result, [str(entry.message) for entry in recorded]


class DeclaredSchemaSetTests(unittest.TestCase):
    def test_supported_set_is_the_ultralytics_family(self):
        self.assertEqual(SUPPORTED_YOLO_SCHEMAS, (YOLO_POSE_RAW, YOLO_POSE_NMS, YOLO_POSE_END2END))

    def test_mmpose_schemas_are_reserved(self):
        self.assertEqual(set(RESERVED_SCHEMAS), {RTMO_MMDEPLOY, RTMPOSE_SIMCC})


class ExplicitSchemaTests(unittest.TestCase):
    """Precedence level 1: the caller named the schema."""

    def test_each_supported_schema_is_accepted_verbatim(self):
        for schema in SUPPORTED_YOLO_SCHEMAS:
            with self.subTest(schema=schema):
                self.assertEqual(resolve(explicit_schema=schema), (schema, None))

    def test_explicit_wins_over_metadata_and_shapes(self):
        resolved = resolve(
            explicit_schema=YOLO_POSE_RAW,
            metadata={"schema": YOLO_POSE_NMS},
            output_shapes={"predictions": (1, MAX_DET, PROCESSED)},
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))

    def test_name_is_normalised(self):
        self.assertEqual(resolve(explicit_schema="  UlTrAlYtIcS_YOLO_Pose_Raw  "),
                         (YOLO_POSE_RAW, None))

    def test_unspecified_values_fall_through(self):
        for blank in (None, "", "  ", "auto", "AUTO", "none"):
            with self.subTest(blank=blank):
                resolved, _ = caught_warnings(
                    lambda blank=blank: resolve(explicit_schema=blank)
                )
                self.assertEqual(resolved, (YOLO_POSE_RAW, None))

    def test_unspecified_really_falls_through_to_metadata(self):
        resolved = resolve(explicit_schema="auto", metadata={"schema": YOLO_POSE_END2END})
        self.assertEqual(resolved, (YOLO_POSE_END2END, None))

    def test_reserved_mmpose_schema_is_rejected_loudly(self):
        for schema in RESERVED_SCHEMAS:
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError) as raised:
                    resolve(explicit_schema=schema)
                message = str(raised.exception)
                self.assertIn("no wiring", message)
                self.assertIn(YOLO_POSE_RAW, message)

    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            resolve(explicit_schema="resnet_heatmap")
        self.assertIn("Unknown pose output schema", str(raised.exception))

    def test_bad_keypoint_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve(num_keypoints=0)
        with self.assertRaises(ValueError):
            resolve(keypoint_dims=0)


class MetadataSchemaTests(unittest.TestCase):
    """Precedence level 2: the artifact's sidecar states the schema."""

    def test_schema_key(self):
        self.assertEqual(resolve(metadata={"schema": YOLO_POSE_NMS}), (YOLO_POSE_NMS, None))

    def test_output_schema_key_is_honoured(self):
        self.assertEqual(resolve(metadata={"output_schema": YOLO_POSE_NMS}), (YOLO_POSE_NMS, None))

    def test_schema_key_wins_over_output_schema_key(self):
        resolved = resolve(metadata={"schema": YOLO_POSE_RAW, "output_schema": YOLO_POSE_NMS})
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))

    def test_end2end_flag(self):
        self.assertEqual(resolve(metadata={"end2end": True}), (YOLO_POSE_END2END, None))

    def test_falsy_end2end_flag_is_ignored(self):
        resolved, messages = caught_warnings(lambda: resolve(metadata={"end2end": False}))
        self.assertEqual(resolved[0], YOLO_POSE_RAW)
        self.assertTrue(messages)

    def test_explicit_schema_beats_end2end_flag(self):
        resolved = resolve(metadata={"end2end": True}, explicit_schema=YOLO_POSE_NMS)
        self.assertEqual(resolved, (YOLO_POSE_NMS, None))

    def test_metadata_wins_over_shapes(self):
        resolved = resolve(
            metadata={"schema": YOLO_POSE_END2END},
            output_shapes={"predictions": (1, ANCHORS, RAW_BASE + 1)},
        )
        self.assertEqual(resolved, (YOLO_POSE_END2END, None))

    def test_reserved_schema_in_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve(metadata={"schema": RTMPOSE_SIMCC})


class ShapeInferenceTests(unittest.TestCase):
    """Precedence level 3: read the layout off the runtime output shape."""

    def test_raw_nc1(self):
        kpt, dims = 1, 3
        base = 4 + kpt * dims
        resolved = resolve(
            output_shapes={"predictions": (1, base + 1, ANCHORS)},
            num_keypoints=kpt,
            keypoint_dims=dims,
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, 1))

    def test_raw_nc3(self):
        resolved = resolve(output_shapes={"predictions": (1, RAW_BASE + 3, ANCHORS)})
        self.assertEqual(resolved, (YOLO_POSE_RAW, 3))

    def test_transposed_raw_is_still_recognised(self):
        kpt, dims = 17, 3
        base = 4 + kpt * dims
        # nc = 1 keeps the channel axis away from the processed channel count,
        # which is what makes the transposed layout unambiguous.
        resolved = resolve(
            output_shapes={"predictions": (1, ANCHORS, base + 1)},
            num_keypoints=kpt,
            keypoint_dims=dims,
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, 1))

    def test_processed_layout_reports_nms_with_a_warning(self):
        # 4 + 2 + K*D == 6 + K*D, so with seventeen 3-D keypoints the channel
        # count alone cannot separate the two layouts -- the axes carrying it
        # can, and only the processed reading puts the channel count last.
        resolved, messages = caught_warnings(
            lambda: resolve(output_shapes={"predictions": (1, MAX_DET, PROCESSED)})
        )
        self.assertEqual(resolved, (YOLO_POSE_NMS, None))
        self.assertEqual(len(messages), 1)
        self.assertIn("indistinguishable by shape", messages[0])

    def test_nc2_raw_is_not_mistaken_for_processed(self):
        resolved, messages = caught_warnings(
            lambda: resolve(output_shapes={"predictions": (1, PROCESSED, ANCHORS)})
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, 2))
        self.assertEqual(messages, [])

    def test_channel_count_axis_must_outnumber_the_count_axis(self):
        # A pose-less tensor: the middle axis is a plain 4-channel box output.
        # Reading the trailing axis as channels would report 8345 classes.
        resolved, messages = caught_warnings(
            lambda: resolve(output_shapes={"output0": (1, 4, ANCHORS)})
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))
        self.assertEqual(len(messages), 1)

    def test_dynamic_dimension_is_skipped(self):
        resolved, messages = caught_warnings(
            lambda: resolve(output_shapes={"predictions": (1, -1, PROCESSED)})
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))
        self.assertEqual(len(messages), 1)

    def test_low_rank_shape_is_skipped(self):
        resolved, messages = caught_warnings(
            lambda: resolve(output_shapes={"predictions": (1, PROCESSED)})
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))
        self.assertEqual(len(messages), 1)

    def test_first_usable_output_wins(self):
        resolved = resolve(
            output_shapes={
                "output0": (1, 4, ANCHORS),          # unusable, skipped
                "output1": (1, RAW_BASE + 7, ANCHORS),
            }
        )
        self.assertEqual(resolved, (YOLO_POSE_RAW, 7))

    def test_shape_may_arrive_as_a_sequence(self):
        resolved = resolve(output_shapes={"predictions": [1, RAW_BASE + 4, ANCHORS]})
        self.assertEqual(resolved, (YOLO_POSE_RAW, 4))


class FallbackTests(unittest.TestCase):
    """Precedence level 4: nothing worked, so say so and pick the default."""

    def test_empty_shapes_warns_and_defaults_to_raw(self):
        resolved, messages = caught_warnings(lambda: resolve(output_shapes={}))
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))
        self.assertEqual(len(messages), 1)
        self.assertIn("--schema", messages[0])

    def test_no_arguments_at_all_defaults_to_raw(self):
        resolved, messages = caught_warnings(lambda: resolve())
        self.assertEqual(resolved, (YOLO_POSE_RAW, None))
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
