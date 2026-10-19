# Technical and hardware considerations

A quick summary of what each stage asks of the machine, and where the bottleneck moves as the
project grows.

## Quick summary

| Stage | Bottleneck | Practical minimum |
|---|---|---|
| labeling | human time | any machine with a display |
| training | GPU compute and video memory | one CUDA GPU, 8 GB+ |
| validation | GPU compute | the training device |
| prediction on a split | GPU compute | the training device |
| live inference | the slowest engine in the chain | CPU is enough for one stream |

## Computer

The toolbox runs on Linux, Windows and macOS. The desktop applications need a display and the
`gui` extra; the CLI runs headless, which is how servers and containers use it.

### Computer hardware

Training scales with the GPU; inference is far more forgiving.

| Workload | Recommendation |
|---|---|
| Label a few hundred frames | any laptop |
| Train an N/S model on a small dataset | one 8 GB CUDA GPU |
| Train an L/X model at 640 px | 16 GB+ CUDA GPU |
| Validate and predict | the same GPU, or a CPU for small runs |
| Live inference, one camera | CPU or an edge device |
| Live inference, several cameras | one GPU, or one device per camera |

Video memory is the binding constraint: batch size times input size times the model's activation
memory. When a run runs out, lower `--batch-size` first, then `--image-size`.

### Camera hardware

Nothing about the camera is prescribed: colour, monochrome, USB, CSI, industrial or phone
footage all work, as long as you can see the landmarks. Two practical points:

- **resolution** — the keypoints need to be visible, not the whole animal; a tight crop at
  640 px beats a wide shot at 4K;
- **frame rate** — for behaviour, the frame rate matters more than the resolution; a 30 fps
  clip with motion blur is worse than a 60 fps clip with less detail.

The inferencer's camera list is a Windows-only convenience (`PyCameraList`); elsewhere pass an
index or a video file.

## Software

| Component | Why it matters |
|---|---|
| PyTorch build | must match the driver's CUDA version |
| engine runtime | only the one you deploy with is required |
| Qt libraries | needed by the desktop applications on Linux |
| OpenCV | comes with the package; the GUI backend affects camera support |

The version matrix and the extras are in [Installing AnimalPoseTracker](../installation.md).

## Measuring your own machine

Two numbers worth knowing before a long run:

```bash
# one forward pass, on a short clip
animalpose-cli infer --config infer.yaml

# epoch duration, on your data
tail -1 results/train/training.jsonl
```

The first tells you whether live inference keeps up; the second tells you whether a training run
finishes tonight.
