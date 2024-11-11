from core.utils.research.model.model.savable import SpinozaModule
from .ssm import SimplifiedSpinozaModule
from .. import ModelCompiler


class SimplifyCompiler(ModelCompiler):

	def compile(self, model: SpinozaModule) -> SimplifiedSpinozaModule:
		simplified_model = SimplifiedSpinozaModule()
		simplified_model.__dict__ = model.__dict__
		simplified_model.forward = model.call
		return simplified_model
