import typing

from lib.utils.logger import Logger
from .models import SampleWeightGenerationModel
from .dataloader import SampleWeightGeneratorDataLoader


class SampleWeightGeneratorTrainer:

	def __init__(
			self,
			model: SampleWeightGenerationModel,
			dataloader: SampleWeightGeneratorDataLoader,
	):

		self.__model = model
		self.__dataloader = dataloader

	def train(self):
		Logger.info("Loading data...")
		X, y = self.__dataloader.load()
		Logger.info(f"Training model(Datasize: {X.shape[0]})...")
		self.__model.fit(X, y)
