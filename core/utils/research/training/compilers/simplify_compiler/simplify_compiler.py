import copy
from core.utils.research.model.model.savable import SpinozaModule
from .ssm import SimplifiedSpinozaModule
from .. import ModelCompiler


class SimplifyCompiler(ModelCompiler):
	def compile(self, model: SpinozaModule) -> SimplifiedSpinozaModule:
		simplified_model = SimplifiedSpinozaModule()

		# Copy properties and methods from model to simplified_model
		for key, value in model.__dict__.items():
			if key not in model._modules:
				# Deep copy to ensure no reference issues
				simplified_model.__dict__[key] = copy.deepcopy(value)

		for name, submodule in model._modules.items():
			if isinstance(submodule, SpinozaModule):
				simplified_model._modules[name] = self.compile(submodule)
			else:
				simplified_model._modules[name] = submodule

		simplified_model.forward = model.call
		return simplified_model
