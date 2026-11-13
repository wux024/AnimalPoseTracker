"""SimCC argmax decoding shared by validation and prediction.

Both paths used to inline this three-line computation. Keeping one copy means a
change to the invalid-score handling cannot silently apply to only one of them.
"""

from typing import Tuple

import numpy as np


def simcc_argmax(pred_x, pred_y, split_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decode ``[B, K, Wx]`` / ``[B, K, Wy]`` simcc maps into crop coordinates.

    Returns ``(coordinates, scores)`` as numpy arrays with shapes ``[B, K, 2]`` and
    ``[B, K]``. Keypoints whose score is non-positive get ``-1 / split_ratio`` so the
    caller can still map them through the affine instead of dropping the row.
    """
    import torch  # local import keeps this module importable without torch

    max_x, coords_x = pred_x.max(dim=-1)
    max_y, coords_y = pred_y.max(dim=-1)
    scores = torch.minimum(max_x, max_y)
    coordinates = torch.stack((coords_x, coords_y), dim=-1).float() / float(split_ratio)
    coordinates[scores <= 0] = -1.0 / float(split_ratio)
    return coordinates.detach().cpu().numpy(), scores.detach().cpu().numpy()


__all__ = ["simcc_argmax"]
