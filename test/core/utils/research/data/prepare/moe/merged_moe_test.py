import unittest

from torch.utils.data import DataLoader

from core.utils.research.data.load import BaseDataset
from core.utils.research.data.prepare.moe import MergedModelOutputExporter
from lib.utils.torch_utils.model_handler import ModelHandler


class MergedMoeTest(unittest.TestCase):

	def setUp(self):

		self.models = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-6-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip"
			]
		]
		print(f"Using {len(self.models)} models")

		self.exporter = MergedModelOutputExporter(
			models=self.models,
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/model_output/cnn-148.cnn-168"
		)
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		self.dataloader = DataLoader(dataset, batch_size=8)

	def test_functionality(self):
		self.exporter.export(self.dataloader)


