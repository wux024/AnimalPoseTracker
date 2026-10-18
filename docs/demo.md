# Code demo

A full run from an empty folder, using one of the dataset profiles that ship with the
package — `Tri-Mouse`, `Fish`, `Horse10`, `LoTE-Web`, `StanfordExtra`, `TopViewMouse` and a
dozen more are listed under `animalposetracker/cfg/datasets`.

```bash
animalpose-cli create --dataset trimouse --workspace projects --name demo --worker me \
    --model AnimalRTPose --scale N --date demo
animalpose-cli train --config projects/demo-me-AnimalRTPose-N-demo/configs/runtime.yaml \
    --epochs 50 --batch-size 8 --device 0
animalpose-cli predict --config projects/demo-me-AnimalRTPose-N-demo/configs/runtime.yaml \
    --weights projects/demo-me-AnimalRTPose-N-demo/results/train/weights/best.pt \
    --source frames/
```

## What it writes

| Path | Contents |
|---|---|
| `configs/` | `project.yaml`, `dataset.yaml`, `model.yaml`, `runtime.yaml`, `cameras.yaml` |
| `results/train/weights/best.pt` | The checkpoint with the best validation score |
| `results/train/results.png` | Loss and metric curves for every epoch |
| `results/val/metrics.json` | AP, AR, PCK, AUC and EPE on the validation split |
| `results/predict/.../predictions.json` | Per-frame keypoints with scores |
| `results/predict/.../*.mp4` | The source video with the skeleton drawn on it |

## Bring your own data

`create --dataset <path/to/dataset.yaml>` accepts any dataset descriptor: keypoint names,
the skeleton, the split folders and the OKS sigmas. The labelling format is YOLO-Pose or
COCO keypoints, both written by `animalpose-annotator`.

## Notebooks

A Colab walkthrough — annotate a few frames, train briefly, watch the skeleton on a video —
is not published yet. Until it is, `animalpose-cli <command> --help` plus
[Getting started](getting_started.md) cover the same ground on a local machine.
