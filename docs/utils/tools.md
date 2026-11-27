# tools/ reference

## convert_legacy_checkpoint.py

Reads a checkpoint written by an older release and writes one the current trainer can
resume. It does not change the weights, only the container.

```bash
python tools/convert_legacy_checkpoint.py old.pt new.pt
```

## demo_clips.py

Cuts the demonstration clips from source footage and writes the GIFs used on the landing
page. It needs the source videos, which are not in the repository.

## screenshots.py

Captures the three windows in their default state and writes the PNGs referenced by the
GUI pages. Run it with the application on screen; it does not fake a window.

## chimpact_tracking_eval.py

CLEAR-MOT (MOTA/MOTP) and IDF1 for ChimpACT, scored video by video. ChimpACT ships one
label file per clip covering every tenth frame; the tracker still runs on every frame so
its motion model sees the whole sequence, but only the annotated frames are scored. Ground-truth
identities come from `bbox_id_ori`. Progress is printed per clip with running MOTA/IDF1.

```bash
python tools/chimpact_tracking_eval.py \
    --weights results/onnx/l/best.onnx \
    --config projects/<project>/configs/dataset.yaml \
    --labels /path/to/ChimpACT_release_v1/labels \
    --videos /path/to/ChimpACT_release_v1/videos_full \
    --tracker bytetrack --output results/tracking_eval/bytetrack
```

Output: `tracking_metrics.json` and `tracking_summary.csv`. The protocol is
`loose-all-clips` — every clip is evaluated, including clips whose frames took part in
training, so the numbers are optimistic; the JSON records this. `--iou` sets the match
threshold (0.5), `--max-clips` / `--max-frames` trim a smoke run, `--self-check` validates
the scorer on synthetic tracks. This script replaces the earlier `tools/tracking_eval.py`.

## dlc_benchmark.py

Writes the data half of a DeepLabCut multi-animal benchmark submission (Lauer et al.,
*Nature Methods* 2022): one JSON object keyed by image name, one instance list per image.
The loader class itself belongs in the benchmark repository, not here.

```bash
python tools/dlc_benchmark.py --predictions results/predict/predictions.json \
    --dataset-yaml configs/dataset.yaml --output trimouse-predictions.json
```

Image names default to `<folder>_<file>`, matching the benchmark's flat
`<video>_<frame>.png` indexing; pass `--keep-basename` for an already-flat layout.
Keypoint names come from `keypoint_names` in the dataset YAML and fall back to `kpt_<i>`;
`--min-score` drops low-confidence instances.

## Adding a tool

A script belongs in `tools/` when it is for maintainers: it is not part of the installed
package, and it may depend on things the runtime does not.
