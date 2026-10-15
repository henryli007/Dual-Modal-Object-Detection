from __future__ import annotations

from collections import OrderedDict

import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.9998) -> None:
        self.decay = decay
        self.shadow = OrderedDict()
        for key, value in model.state_dict().items():
            self.shadow[key] = value.detach().clone()

    @torch.no_grad()
    def update(self, model) -> None:
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> OrderedDict:
        return self.shadow

    def load_state_dict(self, state_dict) -> None:
        self.shadow = OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())
