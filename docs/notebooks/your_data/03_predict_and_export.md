# 3. Predict and export

Run the model trained in [the previous page](02_train_and_validate.md) on new material,
track and filter the sequence, then ship the checkpoint as a deployment artifact.

## Predicting on a folder or a video

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source newfootage/ --conf 0.25
```

`--source` accepts image files, directories, recursive globs, `.txt` manifests and video
files. Results land in `results/predict/`: `predictions.json` with the per-frame keypoints,
plus a rendered video or labelled images beside it. What it does not accept: camera indices
and stream URLs — those are `infer` sources.

## Tracking and filtering the sequence

```bash
animalpose-cli predict --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --source newfootage/clip.mp4 \
    --tracker bytetrack --pose-filter median --filter-window 7
```

On a video source, tracking adds `tracked_predictions.json` (a `track_id` per detection) and
filtering adds `predictions-filter.json` (each track smoothed independently). The rendered
video shows track IDs when tracking is on. The tracker and filter choices — six trackers,
`median` vs `savgol` — are covered in the
[tracking walkthrough](../main_demos/04_tracking_and_filtering.md).

This is also the moment of truth for your labels: scroll the rendered video and watch for
the postures that break. Those frames are what goes back to
[the annotator](01_annotate_and_export.md).

## Exporting to ONNX

```bash
animalpose-cli export --config configs/runtime.yaml \
    --weights results/train/weights/best.pt --format onnx
```

`results/export/` gains `best.onnx` and `best.onnx.json` — the sidecar that records the head,
the input size, the keypoint layout and the preprocessing the graph does not carry. ONNX is
the hub format; OpenVINO, TensorRT and the TensorFlow family are one flag away and are
covered in the [export walkthrough](../extra/01_export_matrix.md).

## Running the artifact with `infer`

```bash
animalpose-cli infer --poser-weights results/export/best.onnx \
    --source newfootage/clip.mp4 --records check.json --output check.mp4
```

`infer` reads no project configuration — the artifact and its sidecar are enough. Running
the exported model on the same clip as step 1 and comparing `check.json` against
`predictions.json` is the real export test: a conversion that moved keypoints shows up here,
not in the exporter's exit code. On a camera or stream the same command runs live
(`--source 0`, `--max-frames 30` for a smoke test); see the
[live inference walkthrough](../extra/02_live_stream.md).

## Where to go next

- [User guide](../../main-workflows/user-guide.md) — the full loop including the refinement
  phase.
- [Driving the pipeline from Python](../extra/04_python_api.md) — the same steps without the
  CLI.
