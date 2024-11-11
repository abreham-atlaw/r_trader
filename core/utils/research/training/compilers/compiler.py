from abc import ABC, abstractmethod

import torch.nn as nn


class ModelCompiler(ABC):

	@abstractmethod
	def compile(self, model: nn.Module) -> nn.Module:
		pass

	def decompile(self, compiled: nn.Module, original: nn.Module) -> nn.Module:
		original.load_state_dict(compiled.state_dict())
		return original
