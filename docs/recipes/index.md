# The AnimalPoseTracker Cookbook

Recipes are short, focused answers to "how do I do X". Each one assumes the install is done and
shows the smallest set of commands or calls that solves the problem.

## What is here

| Recipe | Answers |
|---|---|
| [Public API map](HelperFunctions.md) | which function does what, per module |
| [Bringing your own dataset](OtherData.md) | how to describe your data and where the files go |
| [Inputs and outputs](io.md) | what each stage reads and writes |
| [Editing model graphs](nn.md) | how to change a backbone or a head in YAML |
| [Tracking and multi-view post-processing](post.md) | the tracking, filtering and triangulation API |
| [Batch and multi-camera runs](BatchProcessing.md) | running many sources without a loop in Python |
| [3D association and rendering](three_d_association.md) | index vs epipolar, and the render options |
| [Deployment formats](DeploymentFormats.md) | the export matrix and the sidecar contract |
| [Augmentation and flip tables](flip_and_rotate.md) | how left/right keypoints stay consistent |
| [model.yaml breakdown](pose_cfg_file_breakdown.md) | every key in a model file |
| [Utility scripts](../utils/index.md) | what ships in `tools/` |

## How to read a recipe

Each one is structured the same way: what it solves, the command or call, what it writes, and
what goes wrong. If a recipe grows past a page, it belongs in the
[User guide](../main-workflows/user-guide.md) instead.

## Contributing a recipe

A recipe is worth adding when the same question comes up twice. Keep it to one problem, show the
minimal command, and name the file it writes — the naming is what people search for.
