import os
import typing
import unittest

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import BasicSampleWeightManipulator
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class BasicSampleWeightManipulatorTest(AbstractSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return BasicSampleWeightManipulator(
			shift=3,
			scale=2,
			stretch=10
		)

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [os.path.join(data_path, "w")]
