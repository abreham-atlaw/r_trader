import typing

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


def get_layer_io_saliency(model: nn.Module, X: torch.Tensor, layer: nn.Module) -> torch.Tensor:

	data: torch.Tensor = None

	def pre_layer_hook(_, input):
		nonlocal data
		if not (isinstance(input, typing.Iterable) and isinstance(input[0], torch.Tensor)):
			return input
		data = input[0].detach().requires_grad_(True)
		return (data,)

	def post_layer_hook(_, __, output):
		output.backward(torch.ones_like(output), retain_graph=True)

	pre_hook = layer.register_forward_pre_hook(pre_layer_hook)
	post_hook = layer.register_forward_hook(post_layer_hook)

	model.zero_grad()
	y_hat = model(X)

	for hook in [pre_hook, post_hook]:
		hook.remove()

	if data is None:
		raise ValueError("layer not called in forward pass")

	return torch.mean(data.grad.abs(), dim=0)
