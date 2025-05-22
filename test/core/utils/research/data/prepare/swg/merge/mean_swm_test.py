import typing
import unittest

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.merge import MeanSampleWeightMerger
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class MeanSampleWeightMergerTest(AbstractSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return MeanSampleWeightMerger()

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [
			f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/{i}/test/w"
			for i in [4, 5]
		]
