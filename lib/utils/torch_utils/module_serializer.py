import io
from typing import Dict

import torch.jit
import torch.nn as nn

from lib.network.rest_interface import Serializer


class TorchModuleSerializer(Serializer):

	def __init__(self):
		super().__init__(output_class=nn.Module)

	def serialize(self, data: nn.Module) -> str:
		scripted = torch.jit.script(data)

		buffer = io.BytesIO()
		torch.jit.save(scripted, buffer)
		return buffer.getvalue().hex()

	def deserialize(self, json_: str) -> nn.Module:
		return torch.jit.load(io.BytesIO(bytes.fromhex(json_)))
