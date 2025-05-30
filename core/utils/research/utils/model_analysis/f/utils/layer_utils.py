import typing

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from core.utils.research.model.layers import PassThroughLayer


def replace_layer(model: nn.Module, original_layer: nn.Module, target_layer: nn.Module):
	for module in model.modules():
		for attr_name, child in module.named_children():
			if child is original_layer:
				setattr(module, attr_name, target_layer)
				return
	raise ValueError("Layer not found in model")


def prepare_layer(model: nn.Module, layer: nn.Module):
	target_layer = layer
	if isinstance(layer, torch.jit.ScriptModule):
		target_layer = PassThroughLayer(target_layer)
		replace_layer(model, layer, target_layer)

	return target_layer


def clean_layer(model: nn.Module, original_layer: nn.Module, target_layer: nn.Module, hooks: typing.List[RemovableHandle]):

	if not original_layer is target_layer:
		replace_layer(model, target_layer, original_layer)

	for hook in hooks:
		hook.remove()
