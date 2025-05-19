import unittest
from abc import abstractmethod

from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.data.prepare.swg.training.dataloader import SampleWeightGeneratorDataLoader
from core.utils.research.data.prepare.swg.training.models import SampleWeightGenerationModel
from core.utils.research.data.prepare.swg.training.trainer import SampleWeightGeneratorTrainer


class AbstractModelTest(unittest.TestCase):

	@abstractmethod
	def _init_model(self) -> SampleWeightGenerationModel:
		pass

	def setUp(self):
		self.model = self._init_model()
		self.dataloader = SampleWeightGeneratorDataLoader(
			DataLoader(
				BaseDataset(
					root_dirs=[
						"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
					],
					load_weights=True
				),
				batch_size=8
			),
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)

	def test_fit(self):
		trainer = SampleWeightGeneratorTrainer(
			self.model,
			self.dataloader,
			test_split=0.3
		)
		train_loss = trainer.train()
		test_loss = trainer.evaluate()

		print(f"Train Loss: {train_loss}")
		print(f"Test Loss: {test_loss}")

		self.model.save(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/xgboost_swg_model.xgb")

	def test_predict(self):
		self.test_fit()

		X, y = self.dataloader.load()

		y_hat = self.model.predict(X)

		print(f"Prediction:\n{y_hat}")

	def test_load(self):
		self.test_fit()
		self.model = SampleWeightGenerationModel.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/xgboost_swg_model.xgb")
		self.test_predict()
