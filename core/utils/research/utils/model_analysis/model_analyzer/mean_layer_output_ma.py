import torch

from .layer_output_ma import LayerOutputModelAnalyzer


class MeanLayerOutputModelAnalyzer(LayerOutputModelAnalyzer):

	def _process_output(self, output: torch.Tensor) -> torch.Tensor:
		return torch.unsqueeze(torch.mean(output, dim=0), dim=0)
