# Additional feature walkthroughs

Four walkthroughs that go past the 2D loop: deployment, live inference, 3D, and the Python
API.

## What is here

| Walkthrough | Covers |
|---|---|
| [Export matrix in practice](extra/01_export_matrix.md) | several formats from one checkpoint, and the sidecar |
| [Live inference](extra/02_live_stream.md) | a camera or stream, with records |
| [Full 3D pipeline](extra/03_three_d_pipeline.md) | calibration, association, triangulation |
| [Driving the pipeline from Python](extra/04_python_api.md) | project, tracking and pose3d APIs |

## Before you start

- The export walkthrough needs the extra for each format it reaches — the matrix page lists
  them.
- The live walkthrough needs an artifact and a camera or a video file.
- The 3D walkthrough needs at least two synchronized cameras and a printed ChArUco board.

## Order

They are independent. If you are deploying, read the export walkthrough first; it produces the
artifact the live one consumes.
