import random
import typing
from abc import ABC
from typing import List

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from core.utils.ganno.torch.builder import ModelBuilder
from core.utils.ganno.torch.nnconfig import CNNConfig, ConvLayer, TransformerConfig, ModelConfig
from core.utils.research.training.callbacks import Callback
from core.utils.research.training.trainer import Trainer
from lib.ga import GeneticAlgorithm, Species


class Optimizer(GeneticAlgorithm, ABC):

	def __init__(
			self,
			vocab_size,
			dataset: Dataset,
			test_dataset: Dataset,
			*args,
			population_size=100,
			batch_size=32,
			epochs=10,
			trainer_callbacks: typing.Optional[typing.List[Callback]] = None,
			loader_workers: int = 4,
			**kwargs,

	):
		super().__init__(*args, **kwargs)
		self._population_size = population_size
		self._vocab_size = vocab_size
		self.__builder = ModelBuilder()
		self.__epochs = epochs
		self.__dataloader, self.__test_dataloader = [
			DataLoader(ds, batch_size=batch_size, num_workers=loader_workers)
			for ds in [dataset, test_dataset]
		]
		self.__trainer_callback = trainer_callbacks

	def _evaluate_species(self, species: ModelConfig) -> float:
		model = self.__builder.build(species)
		trainer = Trainer(
			model,
			cls_loss_function=nn.CrossEntropyLoss(),
			reg_loss_function=nn.MSELoss(),
			optimizer=Adam(model.parameters(), lr=1e-3),
			callbacks=self.__trainer_callback
		)
		trainer.train(
			dataloader=self.__dataloader,
			epochs=self.__epochs,
			progress=True
		)

		return 1/trainer.validate(self.__test_dataloader)
