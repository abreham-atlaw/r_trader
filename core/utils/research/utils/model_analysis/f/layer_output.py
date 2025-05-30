import torch
import torch.nn as nn

from core.utils.research.utils.model_analysis.f.utils.layer_utils import prepare_layer, clean_layer


def get_layer_output(model: nn.Module, X: torch.Tensor, layer: nn.Module) -> torch.Tensor:

	target_layer = prepare_layer(model, layer)

	output: torch.Tensor = None

	def forward_hook(_, __, out):
		nonlocal output
		output = out.detach()
		return output

	hook = target_layer.register_forward_hook(forward_hook)

	model.zero_grad()
	y_hat = model(X)

	clean_layer(model, layer, target_layer, [hook])

	if output is None:
		raise ValueError("layer not called in forward pass")

	return output
