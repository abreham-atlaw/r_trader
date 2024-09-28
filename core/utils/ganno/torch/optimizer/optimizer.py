import random
import typing
from abc import ABC
from typing import List

import signal

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from core.utils.ganno.torch.builder import ModelBuilder
from core.utils.ganno.torch.nnconfig import CNNConfig, ConvLayer, TransformerConfig, ModelConfig
from core.utils.research.training.callbacks import Callback
from core.utils.research.training.trainer import Trainer
from lib.ga import GeneticAlgorithm, Species
from lib.utils.logger import Logger


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
			train_timeout: int = None,
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
		self.__train_timeout = train_timeout

	def __set_timeout(self):
		def handle_timeout(*args, **kwargs):
			raise TimeoutException()

		signal.alarm(self.__train_timeout)
		signal.signal(signal.SIGALRM, handler=handle_timeout)

	def _evaluate_species(self, species: ModelConfig) -> float:
		model = self.__builder.build(species)
		trainer = Trainer(
			model,
			callbacks=self.__trainer_callback
		)
		trainer.cls_loss_function = nn.CrossEntropyLoss(),
		trainer.reg_loss_function = nn.MSELoss(),
		trainer.optimizer = Adam(trainer.model.parameters(), lr=1e-3),
		if self.__train_timeout is not None:
			self.__set_timeout()
		try:
			trainer.train(
				dataloader=self.__dataloader,
				epochs=self.__epochs,
				progress=True
			)
		except TimeoutException:
			Logger.info("Training Timeout Reached")

		loss = trainer.validate(self.__test_dataloader)
		if isinstance(loss, typing.Iterable):
			loss = loss[-1]

		return 1/loss


class TimeoutException(Exception):
	pass
