import os
import typing

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.xswg import MomentumXSampleWeightGenerator
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class MomentumXSampleWeightGeneratorTest(AbstractSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return MomentumXSampleWeightGenerator()

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [
			os.path.join(data_path, axis)
			for axis in ["X"]
		]

