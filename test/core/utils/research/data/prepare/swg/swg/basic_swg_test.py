import typing
from abc import ABC

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg import BasicSampleWeightExporter

from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest


class BasicSampleWeightGeneratorTest(AbstractSampleWeightGeneratorTest, ABC):

	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		raise NotImplemented()

	def _init_exporter(self, data_path: str, export_path: str, generator: AbstractSampleWeightGenerator) -> SampleWeightExporter:
		return BasicSampleWeightExporter(
			data_path, export_path, generator
		)
