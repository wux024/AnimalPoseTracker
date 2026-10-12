"""Frame annotation shared by the ``predict`` and ``infer`` command lines.

The two command lines see the world differently -- ``predict`` works with a
detection tuple produced inside torch, ``infer`` with plain dictionaries built
from an exported artifact -- but the marks they want on the frame are the same
four: a bounding box, a label, a skeleton and the keypoints.  That drawing used
to exist twice, once per command, so a change to the palette or the line width
had to be made in two places and the two pictures were free to drift apart.
This module owns it.

Each command keeps a thin adapter that maps its own data shape onto
:func:`draw_instance` and keeps its own look in :data:`PREDICTION_STYLE` or
:data:`INFERENCE_STYLE`.  **Style is data, never a branch**: every visual
difference between the two commands lives in the style objects, so nothing
below contains an ``if predict``.

Callers decide what is drawable and hand over integer pixel coordinates; a
``None`` in ``points`` means "do not draw this keypoint and do not connect any
edge to it".  Keeping that decision outside this module avoids dragging torch or
numpy semantics in here.

Only OpenCV is used, and it is imported lazily inside the drawing call, because
``animalposetracker.inference`` must stay independent of the training stack.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = [
    "Box",
    "Color",
    "Pixel",
    "RenderStyle",
    "PREDICTION_STYLE",
    "INFERENCE_STYLE",
    "draw_instance",
    "pixel_box",
]

Color = Tuple[int, int, int]
Pixel = Optional[Tuple[int, int]]
Box = Optional[Tuple[int, int, int, int]]


def pixel_box(bbox) -> Box:
    """Round ``(x1, y1, x2, y2)`` to integer corners.

    Args:
        bbox: Any four-element sequence of numbers, or a falsy value (``None``,
            an empty list) meaning the source produced no box at all.

    Returns:
        The integer corners, or ``None`` when there is no box to draw.
    """
    if not bbox:
        return None
    values = [int(round(float(value))) for value in bbox]
    return (values[0], values[1], values[2], values[3])


@dataclass(frozen=True)
class RenderStyle:
    """The look of one command line's annotations.

    Attributes:
        box_color: BGR colour of the bounding box, unless ``class_colors`` has an
            entry for the instance's class.
        label_color: BGR colour of the label text, unless ``label_text_color``
            overrides it (a filled label wants dark text on a bright plate).
        skeleton_color: BGR colour of every skeleton edge, unless
            ``skeleton_colors`` has an entry for that edge.
        keypoint_colors: BGR palette cycled by keypoint index.  A single entry
            means every keypoint gets that one colour.
        font_scale: OpenCV Hershey font scale for the label.
        label_offset_y: How far above the box top the label sits.
        label_min_y: Floor for the label anchor, so it never leaves the frame.
        label_thickness_floor: Lower bound for the label stroke width.  ``0``
            means "whatever ``line_width`` says"; ``1`` keeps a hairline label
            readable even when the caller asked for zero-width shapes.
        box_line_width: Stroke width for the box alone.  ``None`` means "use
            ``line_width``", which is what the callers that do not separate the
            two widths want.
        skeleton_line_width: Stroke width for the skeleton edges alone; ``None``
            behaves as ``box_line_width`` does.
        class_colors: BGR palette cycled by class id, for boxes and label plates.
        skeleton_colors: BGR palette cycled by *edge index*, so a skeleton can
            show each limb in its own colour.
        label_fill: Paint a filled rectangle behind the label text.
        label_text_color: BGR colour of the label text when it is filled.
    """

    box_color: Color
    label_color: Color
    skeleton_color: Color
    keypoint_colors: Tuple[Color, ...]
    font_scale: float
    label_offset_y: int
    label_min_y: int
    label_thickness_floor: int
    box_line_width: Optional[int] = None
    skeleton_line_width: Optional[int] = None
    class_colors: Tuple[Color, ...] = ()
    skeleton_colors: Tuple[Color, ...] = ()
    label_fill: bool = False
    label_text_color: Color = (0, 0, 0)


# `predict`: green box and label, cyan skeleton, one colour per keypoint.
PREDICTION_STYLE = RenderStyle(
    box_color=(0, 220, 0),
    label_color=(0, 220, 0),
    skeleton_color=(0, 220, 255),
    keypoint_colors=(
        (0, 200, 255),
        (255, 120, 0),
        (80, 220, 80),
        (220, 80, 220),
        (255, 220, 0),
        (80, 200, 220),
        (200, 120, 255),
        (80, 160, 255),
        (180, 220, 80),
        (220, 180, 80),
        (120, 255, 180),
        (255, 140, 180),
    ),
    font_scale=0.5,
    label_offset_y=5,
    label_min_y=15,
    label_thickness_floor=1,
)

# `infer`: muted olive box, amber skeleton, a single keypoint colour.
INFERENCE_STYLE = RenderStyle(
    box_color=(60, 180, 75),
    label_color=(60, 180, 75),
    skeleton_color=(220, 160, 60),
    keypoint_colors=((80, 80, 230),),
    font_scale=0.6,
    label_offset_y=6,
    label_min_y=0,
    label_thickness_floor=0,
)


def draw_instance(
    frame,
    box: Box,
    label: Optional[str],
    points: Sequence[Pixel],
    skeleton: Sequence[Sequence[int]],
    *,
    style: RenderStyle,
    class_id: Optional[int] = None,
    show_box: bool = True,
    show_label: bool = True,
    show_keypoints: bool = True,
    show_skeleton: bool = True,
    line_width: int = 2,
    point_radius: int = 4,
) -> None:
    """Draw one instance onto ``frame`` in place.

    Args:
        frame: The BGR image to annotate; modified in place.
        box: Integer ``(x1, y1, x2, y2)`` corners, or ``None`` when the source
            produced no box.  The label is anchored to it, so it is also the
            only place a label can be drawn.
        label: Text to place above the box, or ``None``/empty for no label.
        points: One entry per keypoint, in skeleton-index order.  Integer pixel
            coordinates, or ``None`` for a keypoint that must not be drawn.
        skeleton: Keypoint index pairs, 0-based; anything not a pair is skipped.
        style: The palette and typography of the calling command.
        class_id: Index into ``style.class_colors``; ``None`` keeps
            ``style.box_color``.
        show_box: Draw the rectangle.
        show_label: Draw the label.  Independent of ``show_box``.
        show_keypoints: Draw the keypoint dots.
        show_skeleton: Draw the edges.
        line_width: Stroke width for the box, edges and label, unless the style
            overrides the box or the edges separately.
        point_radius: Radius of the keypoint dots.

    Edges are drawn before dots so the dots sit on top of the lines.
    """
    import cv2

    box_color = style.box_color
    if style.class_colors and class_id is not None:
        box_color = style.class_colors[int(class_id) % len(style.class_colors)]
    box_width = line_width if style.box_line_width is None else style.box_line_width
    edge_width = (
        line_width if style.skeleton_line_width is None else style.skeleton_line_width
    )

    if box is not None:
        if show_box:
            cv2.rectangle(
                frame,
                (box[0], box[1]),
                (box[2], box[3]),
                box_color,
                box_width,
            )
        if show_label and label:
            anchor = (box[0], max(style.label_min_y, box[1] - style.label_offset_y))
            if style.label_fill:
                # Measured at the label's own reference size, as the original
                # renderer did: the plate is generous enough for the drawn text.
                (text_width, text_height), _baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (anchor[0], anchor[1] - text_height),
                    (anchor[0] + text_width, anchor[1] + text_height),
                    box_color,
                    cv2.FILLED,
                )
            cv2.putText(
                frame,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                style.label_text_color if style.label_fill else style.label_color,
                max(line_width, style.label_thickness_floor),
                cv2.LINE_AA,
            )

    if show_skeleton:
        edge_colors = style.skeleton_colors
        for index, edge in enumerate(skeleton):
            if len(edge) != 2:
                continue
            left, right = int(edge[0]), int(edge[1])
            if not (0 <= left < len(points) and 0 <= right < len(points)):
                continue
            start, end = points[left], points[right]
            if start is None or end is None:
                continue
            color = edge_colors[index % len(edge_colors)] if edge_colors else style.skeleton_color
            cv2.line(frame, start, end, color, edge_width, cv2.LINE_AA)

    if show_keypoints:
        palette = style.keypoint_colors
        for index, point in enumerate(points):
            if point is None:
                continue
            cv2.circle(
                frame,
                point,
                point_radius,
                palette[index % len(palette)],
                -1,
                cv2.LINE_AA,
            )
