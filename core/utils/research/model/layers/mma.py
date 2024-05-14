from typing import List

import torch
import torch.nn as nn

from .ma import MovingAverage


class MultipleMovingAverages(nn.Module):
    def __init__(self, sizes: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__out_size_decrement = max(sizes) - 1
        self.__mas = nn.ModuleList([MovingAverage(size) for size in sizes])

    def forward(self, inputs, *args, **kwargs):
        out_size = inputs.shape[1] - self.__out_size_decrement
        return torch.stack([
            ma(inputs)[:, -out_size:]
            for ma in self.__mas
        ], dim=2)
