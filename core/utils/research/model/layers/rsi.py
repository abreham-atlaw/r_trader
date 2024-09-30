import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .overlay_indicator import OverlayIndicator
from .delta import Delta
from .sign import SignFilter

class RelativeStrengthIndex(OverlayIndicator):
    def __init__(self, *args, epsilon=1e-7, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.delta = Delta()
        self.gain_filter = SignFilter(1)
        self.loss_filter = SignFilter(-1)

    def _on_time_point(self, inputs: torch.Tensor) -> torch.Tensor:
        percentage = self.delta(inputs) / inputs[:, :, :-1]
        average_gain = torch.mean(
            self.gain_filter(percentage),
            dim=2
        )
        average_loss = torch.mean(
            -1 * self.loss_filter(percentage),
            dim=2
        )
        return 1 - (1 / (1 + average_gain / (average_loss + self.epsilon)))
