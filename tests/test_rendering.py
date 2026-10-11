"""Tests for the frame annotator shared by the `predict` and `infer` commands.

The module under test is deliberately free of torch and numpy, so this file runs
anywhere. OpenCV is replaced by a recorder, which lets the assertions describe
the calls the annotator makes rather than the pixels it produces.
"""

import sys
import unittest
from unittest import mock

from animalposetracker.rendering import (
    INFERENCE_STYLE,
    PREDICTION_STYLE,
    draw_instance,
    pixel_box,
)


class _Frame:
    def __init__(self, tag="frame"):
        self.tag = tag

    def copy(self):
        return _Frame(self.tag)


class _Recorder:
    """Stand-in for cv2 that logs draw calls instead of painting."""

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self):
        self.calls = []

    def rectangle(self, img, pt1, pt2, color, thickness):
        self.calls.append(("rectangle", pt1, pt2, color, thickness))

    def putText(self, img, text, org, font, scale, color, thickness, lineType):
        self.calls.append(("putText", text, org, scale, color, thickness))

    def line(self, img, pt1, pt2, color, thickness, lineType):
        self.calls.append(("line", pt1, pt2, color, thickness))

    def circle(self, img, center, radius, color, thickness, lineType):
        self.calls.append(("circle", center, radius, color))

    def ops(self):
        return [call[0] for call in self.calls]

    def of(self, op):
        return [call for call in self.calls if call[0] == op]


def _draw(**kwargs):
    """Draw one instance against the recorder and return it."""
    recorder = _Recorder()
    with mock.patch.dict(sys.modules, {"cv2": recorder}):
        draw_instance(
            _Frame(),
            kwargs.pop("box", (10, 10, 20, 20)),
            kwargs.pop("label", "mouse 0.90"),
            kwargs.pop("points", [(1, 1), (2, 2)]),
            kwargs.pop("skeleton", [[0, 1]]),
            **kwargs,
        )
    return recorder


class StyleTests(unittest.TestCase):
    def test_styles_keep_each_command_lines_look(self):
        self.assertEqual(PREDICTION_STYLE.box_color, (0, 220, 0))
        self.assertEqual(PREDICTION_STYLE.label_color, (0, 220, 0))
        self.assertEqual(PREDICTION_STYLE.skeleton_color, (0, 220, 255))
        self.assertEqual(len(PREDICTION_STYLE.keypoint_colors), 12)
        self.assertEqual(PREDICTION_STYLE.font_scale, 0.5)
        self.assertEqual(PREDICTION_STYLE.label_offset_y, 5)
        self.assertEqual(PREDICTION_STYLE.label_min_y, 15)
        self.assertEqual(PREDICTION_STYLE.label_thickness_floor, 1)

        self.assertEqual(INFERENCE_STYLE.box_color, (60, 180, 75))
        self.assertEqual(INFERENCE_STYLE.skeleton_color, (220, 160, 60))
        self.assertEqual(INFERENCE_STYLE.keypoint_colors, ((80, 80, 230),))
        self.assertEqual(INFERENCE_STYLE.font_scale, 0.6)
        self.assertEqual(INFERENCE_STYLE.label_offset_y, 6)
        self.assertEqual(INFERENCE_STYLE.label_min_y, 0)
        self.assertEqual(INFERENCE_STYLE.label_thickness_floor, 0)


class PixelBoxTests(unittest.TestCase):
    def test_rounds_each_corner(self):
        self.assertEqual(pixel_box([10.4, 20.5, 60.6, 80.2]), (10, 20, 61, 80))

    def test_missing_box_stays_missing(self):
        self.assertIsNone(pixel_box(None))
        self.assertIsNone(pixel_box([]))


class DrawInstanceTests(unittest.TestCase):
    def test_draws_box_label_edges_then_dots(self):
        recorder = _draw(style=PREDICTION_STYLE)
        self.assertEqual(recorder.ops(), ["rectangle", "putText", "line", "circle", "circle"])

    def test_palette_cycles_by_keypoint_index(self):
        points = [(index, index) for index in range(13)]
        recorder = _draw(style=PREDICTION_STYLE, points=points)
        colors = [call[3] for call in recorder.of("circle")]
        self.assertEqual(colors, list(PREDICTION_STYLE.keypoint_colors) + [PREDICTION_STYLE.keypoint_colors[0]])

    def test_undrawn_keypoint_keeps_its_palette_index(self):
        recorder = _draw(style=PREDICTION_STYLE, points=[(0, 0), None, (2, 2)])
        colors = [call[3] for call in recorder.of("circle")]
        self.assertEqual(colors, [PREDICTION_STYLE.keypoint_colors[0], PREDICTION_STYLE.keypoint_colors[2]])

    def test_single_colour_palette_is_flat(self):
        recorder = _draw(style=INFERENCE_STYLE, points=[(0, 0), (1, 1), (2, 2)])
        self.assertEqual([call[3] for call in recorder.of("circle")], [(80, 80, 230)] * 3)

    def test_edge_needs_both_endpoints(self):
        recorder = _draw(style=PREDICTION_STYLE, points=[(0, 0), None], skeleton=[[0, 1]])
        self.assertEqual(recorder.of("line"), [])
        recorder = _draw(style=PREDICTION_STYLE, points=[(0, 0), (1, 1)], skeleton=[[0, 1]])
        self.assertEqual(len(recorder.of("line")), 1)

    def test_malformed_edge_is_skipped(self):
        for edge in ([0], [0, 1, 2], [0, 9], [-1, 0]):
            with self.subTest(edge=edge):
                recorder = _draw(style=PREDICTION_STYLE, points=[(0, 0), (1, 1)], skeleton=[edge])
                self.assertEqual(recorder.of("line"), [])

    def test_label_is_drawn_even_when_the_box_is_off(self):
        recorder = _draw(style=INFERENCE_STYLE, show_box=False, show_label=True)
        self.assertEqual(recorder.of("rectangle"), [])
        self.assertEqual(len(recorder.of("putText")), 1)

    def test_label_needs_a_box_to_anchor_to(self):
        recorder = _draw(style=INFERENCE_STYLE, box=None, show_label=True)
        self.assertEqual(recorder.of("putText"), [])
        self.assertEqual(recorder.of("rectangle"), [])
        # Skeleton and dots do not depend on the box.
        self.assertEqual(recorder.ops(), ["line", "circle", "circle"])

    def test_empty_label_draws_nothing(self):
        recorder = _draw(style=INFERENCE_STYLE, label="")
        self.assertEqual(recorder.of("putText"), [])

    def test_label_anchor_and_typography_follow_the_style(self):
        high = _draw(style=PREDICTION_STYLE, box=(10, 40, 20, 60), label="m")
        self.assertEqual(high.of("putText")[0][1:5], ("m", (10, 35), 0.5, (0, 220, 0)))
        clipped = _draw(style=PREDICTION_STYLE, box=(10, 12, 20, 60), label="m")
        self.assertEqual(clipped.of("putText")[0][2], (10, 15))
        plain = _draw(style=INFERENCE_STYLE, box=(10, 12, 20, 60), label="m")
        self.assertEqual(plain.of("putText")[0][2], (10, 6))

    def test_label_thickness_floor_keeps_a_hairline_readable(self):
        self.assertEqual(_draw(style=PREDICTION_STYLE, line_width=0).of("putText")[0][5], 1)
        self.assertEqual(_draw(style=INFERENCE_STYLE, line_width=0).of("putText")[0][5], 0)
        self.assertEqual(_draw(style=INFERENCE_STYLE, line_width=3).of("putText")[0][5], 3)


if __name__ == "__main__":
    unittest.main()
