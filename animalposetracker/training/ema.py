"""Exponential moving average of model parameters and floating-point buffers."""

import math
from copy import deepcopy

import torch


class ModelEMA:
    """Maintain a smoothed evaluation copy with the decay ramp used by the reference trainer."""

    def __init__(self, model, decay: float = 0.9999, tau: int = 2000, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.model = deepcopy(model).eval() if self.enabled else model
        self.updates = 0
        self.decay = float(decay)
        self.tau = int(tau)
        if self.enabled:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source_model) -> None:
        if not self.enabled:
            return
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        source_state = source_model.state_dict()
        for name, average in self.model.state_dict().items():
            if average.dtype.is_floating_point:
                average.mul_(decay).add_(source_state[name].detach(), alpha=1.0 - decay)


__all__ = ["ModelEMA"]
