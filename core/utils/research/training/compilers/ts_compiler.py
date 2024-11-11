import torch.jit
import torch.nn as nn
from torch.jit import ScriptModule

from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.training.compilers import ModelCompiler


class TorchScriptCompiler(ModelCompiler):

	def __init__(self):
		pass

	def __generate_sample(self, input_size: torch.Size) -> torch.Tensor:
		return torch.rand((128,) + input_size[1:])

	def compile(self, model: SpinozaModule) -> ScriptModule:
		# sample = self.__generate_sample(model.input_size)
		torch.manual_seed(0)
		model.eval()
		return torch.jit.script(model)
