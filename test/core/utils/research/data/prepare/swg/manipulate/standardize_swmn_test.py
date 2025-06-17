import os
import typing

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest
from test.core.utils.research.data.prepare.swg.manipulate.basic_swmn_test import BasicSampleWeightManipulatorTest


class StandardizeSampleWeightManipulatorTest(AbstractSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return StandardizeSampleWeightManipulator(
			target_std=1,
			target_mean=0,
			current_std=0.94,
			current_mean=0.21
		)

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [
			os.path.join(data_path, "w")
		]
