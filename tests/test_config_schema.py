"""The configuration registry and the defaults file must agree.

These tests pin the definition layer of the config package: key uniqueness,
legal metadata values, and -- the important one -- that every registered
runtime key has exactly one default and every default belongs to exactly one
key. A failure here means schema.py and cfg/runtime_defaults.yaml drifted.
"""

import unittest

from animalposetracker.config import (
    APPLY_MODES,
    FIELDS,
    SCOPES,
    VISIBILITIES,
    default_subtree,
    display_path,
    field_for,
    fields,
    find_by_legacy,
    missing_field_defaults,
    orphan_default_paths,
    runtime_defaults,
    to_cli_name,
)


class RegistryShapeTests(unittest.TestCase):
    def test_every_field_has_legal_metadata(self):
        for field in FIELDS:
            with self.subTest(field=field.path):
                self.assertIn(field.scope, SCOPES)
                self.assertIn(field.apply, APPLY_MODES)
                self.assertIn(field.visibility, VISIBILITIES)
                self.assertTrue(field.note, "every field carries a one-line note")

    def test_keys_are_unique_within_scope_and_group(self):
        seen = set()
        for field in FIELDS:
            address = (field.scope, field.group, field.key)
            self.assertNotIn(address, seen, f"duplicate registry entry: {field.path}")
            seen.add(address)

    def test_the_registry_covers_the_audited_key_set(self):
        # 96 train + 5 common + 21 predict + 11 export = 133 (the concept
        # dictionary, sections 2-5, plus common.selected_weights for the GUI,
        # plus the three one-stage (RTMO) loss keys).
        # Update both the registry and this count together.
        self.assertEqual(len(FIELDS), 133)
        self.assertEqual(len(fields(scope="train")), 96)
        self.assertEqual(len(fields(scope="common")), 5)
        self.assertEqual(len(fields(scope="predict")), 21)
        self.assertEqual(len(fields(scope="export")), 11)

    def test_field_lookup_and_cli_spelling(self):
        field = field_for("batch_size", "train", "models.training")
        self.assertIsNotNone(field)
        self.assertEqual(display_path(field), "train.models.training.batch_size")
        self.assertEqual(to_cli_name("batch_size"), "--batch-size")
        self.assertEqual(display_path(field_for("method", "predict", "pose_filter")),
                         "predict.pose_filter.method")

    def test_legacy_spellings_point_back_at_their_fields(self):
        for legacy, expected in {
            "batch": "batch_size",
            "imgsz": "image_size",
            "val": "validation_enabled",
            "kpt_line": "show_skeletons",
            "prediction_confidence": "confidence",
        }.items():
            with self.subTest(legacy=legacy):
                matches = find_by_legacy(legacy)
                self.assertTrue(matches)
                self.assertIn(expected, [field.key for field in matches])

    def test_type_acceptance_treats_bools_strictly(self):
        integer = field_for("epochs", "train", "models.training")
        floating = field_for("learning_rate", "train", "models.training")

        self.assertTrue(integer.accepts(10))
        self.assertFalse(integer.accepts(True))  # bool is not an int here
        self.assertTrue(floating.accepts(0.01))
        self.assertFalse(floating.accepts(True))  # and not a float either
        self.assertTrue(integer.accepts(None))  # unset is always legal


class DefaultsAlignmentTests(unittest.TestCase):
    def test_every_runtime_field_has_a_default(self):
        self.assertEqual(missing_field_defaults(), [])

    def test_no_default_belongs_to_nobody(self):
        self.assertEqual(orphan_default_paths(), [])

    def test_runtime_defaults_carry_the_agreed_values(self):
        predict = runtime_defaults()["predict"]
        self.assertEqual(predict["confidence"], 0.25)
        self.assertEqual(predict["iou"], 0.7)
        self.assertEqual(predict["pose_filter"]["method"], "none")
        self.assertIsNone(predict["pose_filter"]["confidence_threshold"])
        export = runtime_defaults()["export"]
        self.assertEqual(export["workspace"], 4.0)
        self.assertEqual(export["opset"], 17)
        self.assertIsNone(export["format"])

    def test_subtree_returns_copies(self):
        first = default_subtree("predict")
        first["confidence"] = 0.99
        self.assertEqual(default_subtree("predict")["confidence"], 0.25)

    def test_unknown_scope_is_refused(self):
        with self.assertRaises(ValueError):
            default_subtree("infer")  # infer deliberately has no scope


if __name__ == "__main__":
    unittest.main()
