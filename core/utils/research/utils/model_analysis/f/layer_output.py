import torch
import torch.nn as nn


def get_layer_output(model: nn.Module, X: torch.Tensor, layer: nn.Module) -> torch.Tensor:

	output: torch.Tensor = None

	def forward_hook(_, __, out):
		nonlocal output
		output = out.detach()
		return output

	hook = layer.register_forward_hook(forward_hook)

	model.zero_grad()
	y_hat = model(X)

	hook.remove()

	if output is None:
		raise ValueError("layer not called in forward pass")

	return output
