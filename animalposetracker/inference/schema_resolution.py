"""Decide which ultralytics pose output schema an artifact emits.

The inferencer is a deployment-side decoder: it consumes artifacts produced by
other toolchains and deliberately keeps no dependency on how (or by whom) a
model was trained. Which schema an artifact emits therefore has to be
discoverable at run time, and this module owns that decision in one place as a
pure function, so the CLI and any future embedder behave identically.

Precedence, highest first:

1. an explicit schema (CLI flag / project configuration),
2. the artifact's sidecar metadata,
3. the runtime output shapes reported by the engine,
4. :data:`YOLO_POSE_RAW` as a last resort, with a warning.

Deliberately *not* consulted: metadata embedded inside the model file itself
(for example ONNX ``metadata_props``). Reading it would reintroduce exactly the
exporter-specific coupling this module exists to avoid, so an artifact that
carries no sidecar must be labelled through configuration.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple
import warnings

from animalposetracker.postprocessing.output_decoders import (
    RTMO_MMDEPLOY,
    RTMPOSE_SIMCC,
    SUPPORTED_OUTPUT_SCHEMAS,
    YOLO_POSE_END2END,
    YOLO_POSE_NMS,
    YOLO_POSE_RAW,
)

#: Schemas this module will hand back to the inferencer. RTMPose's SimCC schema is
#: top-down only (it needs an injected detector) and stays reserved here; the
#: one-stage RTMO schema and the ultralytics YOLO family run as-is.
SUPPORTED_YOLO_SCHEMAS: Tuple[str, ...] = (YOLO_POSE_RAW, YOLO_POSE_NMS, YOLO_POSE_END2END)

#: Anything the decoder layer knows but the inferencer cannot yet run.
RESERVED_SCHEMAS: Tuple[str, ...] = tuple(
    schema for schema in SUPPORTED_OUTPUT_SCHEMAS if schema not in SUPPORTED_YOLO_SCHEMAS
)

#: Values for ``explicit_schema`` that mean "nothing was specified".
_UNSPECIFIED = frozenset({"", "auto", "none"})


def resolve_output_schema(
    *,
    explicit_schema: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    output_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    num_keypoints: int,
    keypoint_dims: int = 3,
    allow_topdown: bool = False,
) -> Tuple[str, Optional[int]]:
    """Return ``(schema, inferred_num_classes)`` for one artifact.

    Args:
        explicit_schema: Schema named by the caller (CLI flag or configuration).
            ``None``, ``""``, ``"auto"`` and ``"none"`` all mean "not specified".
        metadata: The artifact's sidecar mapping, as returned by
            :func:`animalposetracker.artifacts.read_artifact_metadata`. May be empty.
        output_shapes: Tensor name to shape, as reported by
            :attr:`InferenceEngine._runtime_output_shapes`. Dynamic dimensions are
            expected to arrive as ``-1``; entries that are unusable are skipped.
        num_keypoints: Keypoint count ``K`` of the artifact.
        keypoint_dims: Dimensions per keypoint ``D`` (2 for ``x, y``, 3 for
            ``x, y, visibility``).

    Returns:
        The resolved schema, plus the class count when the shape inference managed
        to derive one (``None`` otherwise, meaning "ask the caller to work it out").

    Raises:
        ValueError: If an explicitly requested or sidecar-declared schema is not a
            schema the inferencer can run (unknown, or decoder-only and reserved).
    """
    num_keypoints = int(num_keypoints)
    keypoint_dims = int(keypoint_dims)
    if num_keypoints <= 0:
        raise ValueError("num_keypoints must be positive")
    if keypoint_dims <= 0:
        raise ValueError("keypoint_dims must be positive")

    configured = _clean_schema_name(explicit_schema)
    if configured is not None:
        return _validated(configured, allow_topdown=allow_topdown), None

    declared = _declared_schema(metadata or {})
    if declared is not None:
        return _validated(declared, allow_topdown=allow_topdown), None

    inferred = _infer_from_shapes(
        output_shapes or {},
        num_keypoints=num_keypoints,
        keypoint_dims=keypoint_dims,
    )
    if inferred is not None:
        return inferred

    warnings.warn(
        "Could not determine the pose output schema from configuration, artifact "
        f"metadata or runtime output shapes; assuming {YOLO_POSE_RAW!r}. Pass "
        "--schema (or set `schema` in the config yaml) to state it explicitly.",
        stacklevel=2,
    )
    return YOLO_POSE_RAW, None


def _clean_schema_name(value: Any) -> Optional[str]:
    """Normalise a user- or metadata-supplied schema name; ``None`` when unset."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _UNSPECIFIED:
        return None
    return text


def _validated(schema: str, *, allow_topdown: bool = False) -> str:
    """Reject schemas the inferencer cannot actually run, with a precise reason."""
    if schema in SUPPORTED_YOLO_SCHEMAS:
        return schema
    if schema == RTMO_MMDEPLOY:
        # One-stage bottom-up outputs run full-frame: the stream pipeline carries the
        # bottomup preprocessing and the dets/pred_kpts decode, no detector involved.
        return schema
    # RTMPose/SimCC belongs to the top-down branch: it can only run when a detector
    # provider was injected, so the caller unlocks it explicitly.
    if allow_topdown and schema == RTMPOSE_SIMCC:
        return schema
    if schema in SUPPORTED_OUTPUT_SCHEMAS:
        raise ValueError(
            f"Output schema {schema!r} is known to the decoder layer but the inferencer "
            f"has no wiring for it yet; supported here: {', '.join(SUPPORTED_YOLO_SCHEMAS)}"
        )
    raise ValueError(
        f"Unknown pose output schema {schema!r}; supported: {', '.join(SUPPORTED_YOLO_SCHEMAS)}"
    )


def _declared_schema(metadata: Mapping[str, Any]) -> Optional[str]:
    """Schema the artifact's sidecar declares, if any."""
    declared_head = str(metadata.get("head") or "")
    if declared_head == "SimCCHead":
        return RTMPOSE_SIMCC
    if declared_head == "RTMOHead":
        return RTMO_MMDEPLOY
    for key in ("schema", "output_schema"):
        declared = _clean_schema_name(metadata.get(key))
        if declared is not None:
            return declared
    if metadata.get("end2end"):
        return YOLO_POSE_END2END
    return None


def _infer_from_shapes(
    output_shapes: Mapping[str, Sequence[int]],
    *,
    num_keypoints: int,
    keypoint_dims: int,
) -> Optional[Tuple[str, Optional[int]]]:
    """Derive ``(schema, num_classes)`` from the runtime output shapes.

    Two canonical layouts have to be told apart, and they differ in *which axis*
    carries the channels:

    * RAW -- ``(1, 4 + nc + K*D, anchors)``: one column per anchor, channel count
      in the middle. ``anchors`` is large (thousands at 640 px).
    * processed (NMS / end-to-end) -- ``(1, max_det, 6 + K*D)``: one row per
      detection, channel count last. ``max_det`` stays small (a few hundred).

    The trailing axis is tested first, which is what disambiguates ``nc == 2``:
    there ``4 + 2 + K*D`` happens to equal ``6 + K*D``, so a RAW tensor would
    match the processed channel count too -- but only on the *middle* axis, never
    on the trailing one.

    When both axes could be the channel axis (a RAW export with ``anchors`` in
    the middle, or the transposed variant), the axis yielding the smaller
    positive class count wins. The count axis is also required to be at least as
    large as the channel axis: the other axis counts anchors, and an exported
    pose head always has far fewer channels than anchors. That requirement is
    what stops a pose-less tensor -- say ``(1, 4, 8400)`` -- from being read as
    8400-channel output with four anchors.

    NMS and end-to-end outputs are byte-for-byte the same shape and cannot be
    separated here. Both are decoded by the same function, so the choice cannot
    change any result -- it only affects the schema label -- and
    :data:`YOLO_POSE_NMS` is reported with a warning so the label can be pinned
    through configuration when it matters.
    """
    processed_channels = 6 + num_keypoints * keypoint_dims
    raw_base_channels = 4 + num_keypoints * keypoint_dims

    for name, shape in output_shapes.items():
        dimensions = tuple(int(dimension) for dimension in shape)
        if len(dimensions) < 3:
            continue
        middle, last = dimensions[1], dimensions[2]
        if middle < 0 or last < 0:
            continue

        if last == processed_channels:
            warnings.warn(
                f"Output {name!r} has shape {dimensions}, which matches ultralytics' "
                "post-NMS layout; NMS and end-to-end exports are indistinguishable by "
                f"shape, so {YOLO_POSE_NMS!r} is assumed. Both decode identically -- "
                "pass --schema to pin the exact label.",
                stacklevel=2,
            )
            return YOLO_POSE_NMS, None

        candidates = []
        for channel_value, count_value in ((middle, last), (last, middle)):
            if channel_value <= raw_base_channels:
                continue
            if count_value < channel_value:
                continue
            candidates.append(channel_value - raw_base_channels)
        if candidates:
            return YOLO_POSE_RAW, min(candidates)

    return None


__all__ = ["SUPPORTED_YOLO_SCHEMAS", "RESERVED_SCHEMAS", "resolve_output_schema"]
