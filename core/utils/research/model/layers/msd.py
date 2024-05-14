import torch

from .overlay_indicator import OverlayIndicator


class MovingStandardDeviation(OverlayIndicator):
    def __init__(self, *args, **kwargs):
        super(MovingStandardDeviation, self).__init__(*args, **kwargs)

    def _on_time_point(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            torch.sum(
                torch.pow(
                    inputs - torch.mean(inputs, dim=1, keepdim=True),
                    2
                )/inputs.shape[1],
                dim=1
            )
        )
