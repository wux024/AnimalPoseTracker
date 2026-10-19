# Filter settings

Smoothing runs over a whole sequence, per track, after the run ends. The settings decide how much
of the signal survives.

## Methods

| Method | Good for | Cost |
|---|---|---|
| `median` | spike removal — a keypoint that jumps for one frame | fast, no phase shift |
| `savgol` | smooth drift — a jitter that follows the motion | preserves peaks better than a moving average |

`median` is the safer default: it removes outliers without changing the shape of the motion.
`savgol` needs a window and a polynomial order, and behaves badly when the window is longer than
the shortest real movement.

## Settings

| Setting | Effect |
|---|---|
| window | frames averaged; odd numbers keep the output aligned with the input |
| order | polynomial order for `savgol`; must be below the window |
| confidence | points below it are treated as missing, then interpolated across |
| max gap | longest run of missing frames that is interpolated; longer runs stay holes |

## GUI vs CLI

| GUI | CLI |
|---|---|
| **Filter** tick box | `--pose-filter median\|savgol` |
| method dropdown | `--pose-filter` |
| window / order | `--filter-window`, `--filter-polyorder` |
| confidence | `--filter-confidence` |
| gap | `--filter-max-gap` |

Both refuse to filter without tracking: the filter works per track, and without ids there is no
track to smooth. The GUI disables the controls; the CLI rejects the combination.

## What filtering does not do

- It does not recover a keypoint the model never predicted — a gap longer than `max gap` stays a
  gap.
- It does not fix a box. Tracking and filtering operate on keypoints and ids; a wrongly sized box
  stays a wrongly sized box.
- It is not a substitute for more labels. Smoothing hides jitter; it does not teach the model a
  landmark it never learned.
