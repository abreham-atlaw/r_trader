import typing

from torch import nn as nn

from core.utils.research.training.compilers import ModelCompiler


class PipelineCompiler(ModelCompiler):

	def __init__(self, compilers: typing.List[ModelCompiler]):
		self.__compilers = compilers

	def compile(self, model: nn.Module) -> nn.Module:
		for compiler in self.__compilers:
			model = compiler.compile(model)
		return model
