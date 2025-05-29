import typing

import torch.nn as nn


class LayerUtils:

	@staticmethod
	def get_layers(model: nn.Module, skip_identity: bool = True) -> typing.Dict[str, nn.Module]:
		return {
			name: layer
			for name, layer in model.named_modules()
			if not isinstance(layer, nn.Identity) or not skip_identity
		}
