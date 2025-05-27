import os

import torch

from .layer_output_ma import LayerOutputModelAnalyzer


class MeanLayerOutputModelAnalyzer(LayerOutputModelAnalyzer):

	def _generate_output_path(self, export_path: str, name: str) -> str:
		return os.path.join(export_path, f"mean_layer_output-{name}.npy")

	def _process_output(self, output: torch.Tensor) -> torch.Tensor:
		return torch.unsqueeze(torch.mean(output, dim=0), dim=0)
