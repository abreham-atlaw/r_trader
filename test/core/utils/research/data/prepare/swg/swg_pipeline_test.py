import os
import typing

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import BasicSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg import VolatilitySampleWeightGenerator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class SampleWeightGeneratorPipelineTest(AbstractSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return SampleWeightGeneratorPipeline(
			generators=[
				VolatilitySampleWeightGenerator(
					bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
				),
				BasicSampleWeightManipulator(scale=2)
			]
		)

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		return [
			os.path.join(data_path, axis)
			for axis in ["X", "y"]
		]
