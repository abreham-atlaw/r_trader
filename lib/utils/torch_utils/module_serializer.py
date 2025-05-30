import io

import torch.jit
import torch.nn as nn

from core.utils.research.model.layers import PassThroughLayer
from lib.network.rest_interface import Serializer


class TorchModuleSerializer(Serializer):

	def __init__(self, passthrough_wrap: bool = False):
		super().__init__(output_class=nn.Module)
		self.__passthrough_wrap = passthrough_wrap

	def serialize(self, layer: nn.Module) -> str:
		if isinstance(layer, PassThroughLayer) and self.__passthrough_wrap:
			layer = layer.layer
		scripted = torch.jit.script(layer)

		buffer = io.BytesIO()
		torch.jit.save(scripted, buffer)
		return buffer.getvalue().hex()

	def deserialize(self, json_: str) -> nn.Module:
		layer = torch.jit.load(io.BytesIO(bytes.fromhex(json_)))
		if self.__passthrough_wrap:
			layer = PassThroughLayer(layer)
		return layer
