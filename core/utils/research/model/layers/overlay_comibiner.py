from typing import List, Tuple

import torch
import torch.nn as nn


class OverlaysCombiner(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, *args, **kwargs):
        if isinstance(inputs, torch.Tensor) and len(inputs.shape) < 3:
            return inputs.unsqueeze(2)
        min_size = min([overlay.shape[1] for overlay in inputs])
        return torch.stack([
            overlay[:, -min_size:]
            for overlay in inputs
        ], dim=2)

