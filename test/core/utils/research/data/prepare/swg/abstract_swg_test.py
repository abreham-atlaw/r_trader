import os
import typing
import unittest
from abc import abstractmethod, ABC

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator


class AbstractSampleWeightGeneratorTest(unittest.TestCase, ABC):

	@abstractmethod
	def _init_generator(self) -> AbstractSampleWeightGenerator:
		pass

	@abstractmethod
	def _get_input_paths(self, data_path: str) -> typing.List[str]:
		pass

	def _init_exporter(self, data_path: str, export_path: str, generator: AbstractSampleWeightGenerator) -> SampleWeightExporter:
		return SampleWeightExporter(
			input_paths=self._get_input_paths(data_path),
			export_path=export_path,
			generator=generator
		)

	def setUp(self):
		self.generator = self._init_generator()
		self.data_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/5/train"
		self.export_path = os.path.join(self.data_path, "w")
		self.exporter = self._init_exporter(self.data_path, self.export_path, self.generator)

	def test_functionality(self):
		self.exporter.start()
