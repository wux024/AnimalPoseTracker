# Project workspace GUI

`animalposetracker` is the window that drives the whole 2D workflow: create or load a
project, extract frames, train, evaluate, predict, export, and — for 3D projects — calibrate
and triangulate. It writes exactly the files the CLI writes, so a run started here can be
continued in a terminal and vice versa.

## Getting started

```bash
conda activate animalposetracker
animalposetracker
```

The window opens on the project list. From there:

- **Create** builds a project from a descriptor or your own dataset, the same call as
  `animalpose-cli create`;
- **Load** opens an existing project directory;
- the tabs then follow the workflow: **Manage**, **Train**, **Evaluate**, **Predict**,
  **Export**, and **3D Reconstruction** when the project has the flag.

The status bar at the bottom reports the environment (Python, PyTorch, CUDA) the window is
running in, which is the first thing to check when a run fails immediately.

## User guide

Each stage has its own page, in the order you meet it:

| Page | Covers |
|---|---|
| [Setting up a new project](../beginner-guides/beginners-guide.md) | first launch, create vs load, what the form asks for |
| [Managing a project](../beginner-guides/manage-project.md) | project files, frame extraction, splits, configuration edits |
| [Labeling poses](../beginner-guides/labeling.md) | the annotator window, boxes, keypoints, saving |
| [Training and evaluation](../beginner-guides/Training-Evaluation.md) | the training form, curves, validation metrics |
| [Video analysis and export](../beginner-guides/video-analysis.md) | predict, export, the inferencer, the 3D tab |

Two other windows have their own pages:

- [Annotator GUI](annotator_GUI.md) — boxes, classes, keypoints, export.
- [Inferencer](../live/gui/index.md) — an exported artifact on a camera, video or stream.

## Video demos

Short recordings of the same steps, on the built-in datasets:

### How to launch and run the Project Manager GUI

https://github.com/user-attachments/assets/156b6e16-5660-4c81-b14b-73120b67417a

### Creating a project, extracting frames and labeling

https://github.com/user-attachments/assets/c48b6511-7979-4a84-ac59-b9ba4d6edebe

### Training, evaluating and predicting

https://github.com/user-attachments/assets/79636a6e-3813-40f1-8595-06040e2fe178

### Running an exported model in the inferencer

https://github.com/user-attachments/assets/db726dd8-a092-40e3-94ea-e3d423973a03

Each clip is a demonstration, not a tutorial; the five pages above describe the same steps
with the settings that matter.
