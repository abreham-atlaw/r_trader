from typing import Dict

from core.utils.research.eval.mlpl_evaluator.mlpl_optimization.ga.mlpl_species import MLPLSpecies
from lib.network.rest_interface import Serializer


class MLPLSpeciesSerializer(Serializer):

	def __init__(self):
		super().__init__(MLPLSpecies)

	def serialize(self, data: MLPLSpecies) -> Dict:
		return {
			"models": data.models
		}

	def deserialize(self, json_: Dict) -> MLPLSpecies:
		return MLPLSpecies(
			models=json_["models"]
		)
