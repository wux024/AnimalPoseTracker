"""--batch-size auto: argument parsing."""

import unittest

from animalposetracker.training.batch_probe import parse_batch_size


class ParseBatchSizeTests(unittest.TestCase):
    def test_auto_spellings(self):
        self.assertEqual(parse_batch_size("auto"), "auto")
        self.assertEqual(parse_batch_size("AUTO"), "auto")
        self.assertEqual(parse_batch_size(" auto "), "auto")
        self.assertEqual(parse_batch_size("-1"), "auto")
        self.assertEqual(parse_batch_size(-1), "auto")

    def test_concrete_integers(self):
        self.assertEqual(parse_batch_size(32), 32)
        self.assertEqual(parse_batch_size("64"), 64)
        self.assertEqual(parse_batch_size(1), 1)

    def test_rejects_garbage(self):
        for bad in ("big", "abc", True, None, ""):
            with self.assertRaises(ValueError):
                parse_batch_size(bad)


if __name__ == "__main__":
    unittest.main()
