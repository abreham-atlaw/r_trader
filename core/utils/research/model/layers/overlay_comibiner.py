import typing
from typing import List, Tuple

import torch
import torch.nn as nn


class OverlaysCombiner(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs: typing.List[torch.Tensor]) -> torch.Tensor:
        min_size = min([overlay.shape[1] for overlay in inputs])
        return torch.stack([
            overlay[:, -min_size:]
            for overlay in inputs
        ], dim=1)

