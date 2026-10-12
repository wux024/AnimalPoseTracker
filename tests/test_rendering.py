"""Tests for the frame annotator shared by the `predict` and `infer` commands.

The module under test is deliberately free of torch and numpy, so this file runs
anywhere. OpenCV is replaced by a recorder, which lets the assertions describe
the calls the annotator makes rather than the pixels it produces.
"""

import sys
import unittest
from dataclasses import replace
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
    FILLED = -1

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

    def getTextSize(self, text, font, scale, thickness):
        return (4 * len(text), 8), 2

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


class PerClassAndPerEdgeColourTests(unittest.TestCase):
    """The inferencer page shows one colour per class, one per skeleton edge, and a filled
    label. Those are style data: the neutral renderer stays free of "if this caller"."""

    CLASSES = ((11, 22, 33), (44, 55, 66))
    EDGES = ((1, 2, 3), (4, 5, 6))

    def test_box_colour_follows_the_class_id(self):
        style = replace(PREDICTION_STYLE, class_colors=self.CLASSES)
        first = _draw(style=style, class_id=0).of("rectangle")[0]
        second = _draw(style=style, class_id=1).of("rectangle")[0]
        self.assertEqual(first[3], self.CLASSES[0])
        self.assertEqual(second[3], self.CLASSES[1])

    def test_without_class_colours_the_plain_box_colour_is_used(self):
        style = replace(PREDICTION_STYLE, class_colors=self.CLASSES)
        self.assertEqual(
            _draw(style=style, class_id=None).of("rectangle")[0][3], PREDICTION_STYLE.box_color
        )

    def test_edge_colours_cycle_by_edge_index(self):
        style = replace(PREDICTION_STYLE, skeleton_colors=self.EDGES)
        recorder = _draw(
            style=style,
            points=[(0, 0), (1, 1)],
            skeleton=[[0, 1], [0, 1], [0, 1]],
        )
        self.assertEqual(
            [call[3] for call in recorder.of("line")],
            [self.EDGES[0], self.EDGES[1], self.EDGES[0]],
        )

    def test_without_edge_colours_the_plain_skeleton_colour_is_used(self):
        style = replace(PREDICTION_STYLE, skeleton_colors=self.EDGES)
        recorder = _draw(style=replace(style, skeleton_colors=()), skeleton=[[0, 1]])
        self.assertEqual(recorder.of("line")[0][3], PREDICTION_STYLE.skeleton_color)

    def test_box_and_edge_widths_are_independent(self):
        style = replace(PREDICTION_STYLE, box_line_width=5, skeleton_line_width=1)
        recorder = _draw(style=style, line_width=3)
        self.assertEqual(recorder.of("rectangle")[0][4], 5)
        self.assertEqual(recorder.of("line")[0][4], 1)
        # The label still follows the caller's line width.
        self.assertEqual(recorder.of("putText")[0][5], max(3, PREDICTION_STYLE.label_thickness_floor))

    def test_a_width_of_none_keeps_the_caller_line_width(self):
        recorder = _draw(style=PREDICTION_STYLE, line_width=7)
        self.assertEqual(recorder.of("rectangle")[0][4], 7)
        self.assertEqual(recorder.of("line")[0][4], 7)

    def test_filled_label_paints_a_plate_before_the_text(self):
        style = replace(
            PREDICTION_STYLE, class_colors=self.CLASSES, label_fill=True, label_text_color=(0, 0, 0)
        )
        recorder = _draw(style=style, class_id=1, label="dog 0.90", box=(10, 40, 20, 60))

        self.assertEqual(recorder.ops(), ["rectangle", "rectangle", "putText", "line", "circle", "circle"])
        plate = recorder.of("rectangle")[1]
        self.assertEqual(plate[3], self.CLASSES[1])
        self.assertEqual(plate[4], _Recorder.FILLED)
        # A plate is generous around the anchor: from y - h to y + h.
        self.assertEqual(plate[1], (10, 40 - 5 - 8))
        self.assertEqual(plate[2], (10 + 4 * len("dog 0.90"), 40 - 5 + 8))
        self.assertEqual(recorder.of("putText")[0][4], (0, 0, 0))

    def test_without_label_fill_the_text_keeps_its_colour(self):
        recorder = _draw(style=PREDICTION_STYLE, label="m")
        self.assertEqual(len(recorder.of("rectangle")), 1)  # the box only
        self.assertEqual(recorder.of("putText")[0][4], PREDICTION_STYLE.label_color)


if __name__ == "__main__":
    unittest.main()
