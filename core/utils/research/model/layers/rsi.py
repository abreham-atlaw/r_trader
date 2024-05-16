import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .overlay_indicator import OverlayIndicator
from .delta import Delta
from .sign import SignFilter

class RelativeStrengthIndex(OverlayIndicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__delta = Delta()
        self.__gain_filter = SignFilter(1)
        self.__loss_filter = SignFilter(-1)

    def _on_time_point(self, inputs: torch.Tensor) -> torch.Tensor:
        percentage = self.__delta(inputs) / inputs[:, :, :-1]
        average_gain = torch.mean(
            self.__gain_filter(percentage),
            dim=2
        )
        average_loss = torch.mean(
            -1 * self.__loss_filter(percentage),
            dim=2
        )
        return 1 - (1 / (1 + average_gain / average_loss))
