import unittest

from torch.utils.data import DataLoader

from core.utils.research.data.load import BaseDataset
from core.utils.research.data.prepare import ModelOutputExporter
from lib.utils.torch_utils.model_handler import ModelHandler


class ModelOutputExporterTest(unittest.TestCase):

	def setUp(self):
		self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-192-cum-0-it-4-tot.zip")
		self.exporter = ModelOutputExporter(
			model=self.model,
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/model_output/cnn-192"
		)
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		self.dataloader = DataLoader(dataset, batch_size=8)

	def test_functionality(self):
		self.exporter.export(self.dataloader)


