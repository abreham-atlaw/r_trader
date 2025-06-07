import typing

import torch
import torch.nn as nn


class LayerUtils:

	@staticmethod
	def get_layers(model: nn.Module, skip_identity: bool = True) -> typing.Dict[str, nn.Module]:
		return {
			name: layer
			for name, layer in model.named_modules()
			if not isinstance(layer, nn.Identity) or not skip_identity
		}

	@staticmethod
	def get_layer_weights(layer: nn.Module) -> typing.Dict[str, torch.Tensor]:
		return {
			name: weight
			for name, weight in layer.named_parameters(recurse=False)
		}


