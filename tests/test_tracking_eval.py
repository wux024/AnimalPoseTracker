"""The tracking evaluation tool ships its own self-check; this keeps it running."""

import importlib.util
import unittest
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "tracking_eval.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("tracking_eval_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackingEvalTests(unittest.TestCase):
    def test_self_check_passes(self):
        module = load_tool()
        self.assertEqual(module._self_check(), 0)

    def test_idf1_is_one_when_every_gt_is_matched_once(self):
        module = load_tool()
        idf1, idp, idr, idtp = module.idf1_metrics({("a", 1): 5, ("b", 2): 5}, 10, 10)

        self.assertAlmostEqual(idf1, 1.0)
        self.assertAlmostEqual(idp, 1.0)
        self.assertAlmostEqual(idr, 1.0)
        self.assertEqual(idtp, 10)

    def test_idf1_drops_when_ids_are_fragmented(self):
        module = load_tool()
        # Each GT id is split across two predicted ids: half the matches are still correct.
        idf1, _idp, _idr, idtp = module.idf1_metrics(
            {("a", 1): 3, ("a", 2): 2, ("b", 3): 3, ("b", 4): 2}, 10, 10
        )

        self.assertEqual(idtp, 6)
        self.assertLess(idf1, 1.0)

    def test_natural_key_orders_numeric_frames(self):
        module = load_tool()
        names = ["000010.jpg", "000002.jpg", "000001.jpg"]
        self.assertEqual(
            sorted(names, key=module.natural_key),
            ["000001.jpg", "000002.jpg", "000010.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
