from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class OverlayIndicator(nn.Module, ABC):

    def __init__(self, window_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__window_size = window_size

    def get_window_size(self) -> int:
        return self.__window_size

    @abstractmethod
    def _on_time_point(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, inputs, *args, **kwargs):
        windows = inputs.unfold(dimension=1, size=self.__window_size, step=1)
        output = self._on_time_point(windows)
        return output
