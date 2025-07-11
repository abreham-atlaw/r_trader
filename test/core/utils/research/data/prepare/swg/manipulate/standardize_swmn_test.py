import os
import typing

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest
from test.core.utils.research.data.prepare.swg.manipulate.basic_swmn_test import BasicSampleWeightManipulatorTest


class StandardizeSampleWeightManipulatorTest(AbstractSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return StandardizeSampleWeightManipulator(
			target_std=0.3,
			target_mean=1,
			current_std=0.0327,
			current_mean=1.09,
			min_weights=0
		)

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [
			os.path.join(data_path, "w")
		]
