"""SimCC loss adapter used by the shared training loop."""

from typing import Dict

import torch
import torch.nn.functional as F


class SimCCKLLoss:
    """MMPose ``KLDiscretLoss`` defaults for the AnimalViTPose SimCC head."""

    # MMPose sums the per-instance KL terms and divides by K, not by batch size.
    loss_is_batch_sum = True

    def __init__(self, keypoint_count: int, beta: float = 1.0) -> None:
        self.keypoint_count = int(keypoint_count)
        self.beta = float(beta)
        if self.keypoint_count < 1 or self.beta <= 0:
            raise ValueError("keypoint_count and KL beta must be positive")

    def __call__(self, predictions, targets: Dict[str, torch.Tensor]):
        if not isinstance(predictions, (tuple, list)) or len(predictions) != 2:
            raise TypeError("SimCCHead must return the x and y coordinate logits")
        pred_x, pred_y = predictions
        target_x = targets["simcc_x"]
        target_y = targets["simcc_y"]
        weights = targets["keypoint_weights"].reshape(-1)
        if pred_x.shape != target_x.shape or pred_y.shape != target_y.shape:
            raise ValueError(
                "SimCC prediction/target shapes differ: "
                f"x={tuple(pred_x.shape)}/{tuple(target_x.shape)}, "
                f"y={tuple(pred_y.shape)}/{tuple(target_y.shape)}"
            )
        if pred_x.shape[1] != self.keypoint_count or pred_y.shape[1] != self.keypoint_count:
            raise ValueError("SimCC output keypoint count does not match the dataset")

        def axis_loss(prediction, target):
            logits = prediction.reshape(-1, prediction.shape[-1]) * self.beta
            labels = target.reshape(-1, target.shape[-1])
            log_probability = F.log_softmax(logits, dim=1)
            divergence = F.kl_div(log_probability, labels, reduction="none").mean(dim=1)
            return (divergence * weights).sum() / self.keypoint_count

        loss_x = axis_loss(pred_x, target_x)
        loss_y = axis_loss(pred_y, target_y)
        return {
            "loss": loss_x + loss_y,
            "loss_simcc_x": loss_x,
            "loss_simcc_y": loss_y,
        }
