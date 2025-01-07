from typing import List

from core.di import ServiceProvider
from core.utils.research.model.model.savable import SpinozaModule
from lib.ga import GeneticAlgorithm
from lib.ga.choice_utils import ChoiceUtils
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.torch_utils.model_handler import ModelHandler

from .mlpl_species import MLPLSpecies
from ... import MLPLEvaluator


class MLPLGAOptimizer(GeneticAlgorithm):

	def __init__(
			self,
			models: List[str],
			population_size: int,
			evaluator: MLPLEvaluator
	):
		super().__init__()
		self.__models = models
		self.__population_size = population_size
		self.__evaluator = evaluator
		self.__fs = ServiceProvider.provide_file_storage()

	@CacheDecorators.cached_method()
	def __get_model(self, name: str) -> SpinozaModule:
		self.__fs.download(name)
		return ModelHandler.load(name)

	def _generate_initial_generation(self) -> List[MLPLSpecies]:
		return [
			MLPLSpecies(
				models=ChoiceUtils.list_select(
					a=self.__models,
					b=self.__models
				)
			)
			for _ in range(self.__population_size)
		]

	def _evaluate_species(self, species: MLPLSpecies) -> float:
		return 1/self.__evaluator.evaluate([
			self.__get_model(model)
			for model in species.models
		])
