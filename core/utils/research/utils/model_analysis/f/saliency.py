import torch
import torch.nn as nn


def get_input_saliency(model: nn.Module, X: torch.Tensor):

	X = X.clone().detach().requires_grad_(True)
	y_hat = model(X)
	y_hat.backward(torch.ones_like(y_hat))
	saliency = torch.mean(X.grad.abs(), dim=0)

	return saliency


def get_layer_saliency(model: nn.Module, X: torch.Tensor, layer: nn.Module):

	activation: torch.Tensor = None

	def forward_hook(_, __, output):
		nonlocal activation
		activation = output.detach().requires_grad_(True)
		return activation

	hook = layer.register_forward_hook(forward_hook)

	model.zero_grad()
	y_hat = model(X)

	y_hat.backward(torch.ones_like(y_hat))

	hook.remove()

	if activation is None:
		raise ValueError("layer not called in forward pass")

	return torch.mean(activation.grad.abs(), dim=0)
