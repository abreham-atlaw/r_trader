import typing

import torch
import torch.nn as nn

from core.utils.research.utils.model_analysis.f.utils.layer_utils import prepare_layer, clean_layer


def get_input_saliency(model: nn.Module, X: torch.Tensor):

	X = X.clone().detach().requires_grad_(True)
	y_hat = model(X)
	y_hat.backward(torch.ones_like(y_hat))
	saliency = torch.mean(X.grad.abs(), dim=0)

	return saliency


def get_layer_saliency(model: nn.Module, X: torch.Tensor, layer: nn.Module):

	target_layer = prepare_layer(model, layer)

	activation: torch.Tensor = None

	def forward_hook(_, __, output):
		nonlocal activation
		activation = output.detach().requires_grad_(True)
		return activation

	hook = target_layer.register_forward_hook(forward_hook)

	model.zero_grad()
	y_hat = model(X)

	y_hat.backward(torch.ones_like(y_hat))

	clean_layer(model, layer, target_layer, [hook])

	if activation is None or activation.grad is None:
		raise ValueError("layer not called in forward pass")

	return torch.mean(activation.grad.abs(), dim=0)


def get_layer_io_saliency(model: nn.Module, X: torch.Tensor, layer: nn.Module) -> torch.Tensor:

	target_layer = prepare_layer(model, layer)

	data: torch.Tensor = None

	def pre_layer_hook(_, input):
		nonlocal data
		if not (isinstance(input, typing.Iterable) and isinstance(input[0], torch.Tensor)):
			raise ValueError(f"Input type not known. Type: {type(input)}. Input: {input}")
		data = input[0].detach().requires_grad_(True)
		return (data,)

	def post_layer_hook(_, __, output):
		output.backward(torch.ones_like(output), retain_graph=True)

	pre_hook = target_layer.register_forward_pre_hook(pre_layer_hook)
	post_hook = target_layer.register_forward_hook(post_layer_hook)

	model.zero_grad()
	try:
		y_hat = model(X)
	finally:
		clean_layer(model, layer, target_layer, [pre_hook, post_hook])

	if data is None:
		raise ValueError("layer not called in forward pass")

	return torch.mean(data.grad.abs(), dim=0)
