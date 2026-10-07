#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: tal.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2026/9/16
Version: 1.0

Overview:
    Minimal task-aligned helpers: anchor generation and distance/box conversion.

    Only the two functions needed for pose estimation are kept here; the rest of the
    upstream task-aligned assigner (training-only logic) stays on the training side.

Notes:
    - Depends on torch only, no ultralytics / mmcv / mmengine.

Revision History:
    - [2026/9/16] wux024: Initial file creation
"""

import torch


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points and stride tensors from multi-scale feature maps.

    Args:
        feats (list[torch.Tensor]): Feature maps per scale; shape[2:] gives h, w
        strides (torch.Tensor): Stride per scale
        grid_cell_offset (float): Grid cell center offset

    Returns:
        anchor_points (torch.Tensor): (N, 2), anchors of all scales concatenated
        stride_tensor (torch.Tensor): (N, 1), matching strides
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Decode (left, top, right, bottom) distances into boxes.

    Args:
        distance (torch.Tensor): (..., 4), ltrb distances
        anchor_points (torch.Tensor): (..., 2), anchor centers
        xywh (bool): True returns xywh, False returns xyxy
        dim (int): Channel dimension
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


__all__ = ["make_anchors", "dist2bbox"]
