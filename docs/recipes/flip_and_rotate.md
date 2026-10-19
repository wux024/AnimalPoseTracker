# Augmentation and flip tables

## The flip table

- `flip_idx` maps each keypoint to its left/right counterpart
- a wrong table teaches the model that left and right are the same landmark
- keypoints without a counterpart map to themselves

## Geometric augmentation

- mosaic and mixup change the composition, not the labels
- rotation and scale are bounded by the instance size
- colour jitter is applied after geometry, never before

## What to disable

- flip augmentation for a task where left/right is the label
- mosaic for dense scenes where instances overlap heavily
- any augmentation that changes the keypoint order
