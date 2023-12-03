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
			**kwargs,

	):
		super().__init__(*args, **kwargs)
		self._population_size = population_size
		self._vocab_size = vocab_size
		self.__builder = ModelBuilder()
		self.__epochs = epochs
		self.__dataloader, self.__test_dataloader = [
			DataLoader(ds, batch_size=batch_size)
			for ds in [dataset, test_dataset]
		]
		self.__trainer_callback = trainer_callbacks

	def _evaluate_species(self, species: ModelConfig) -> float:
		model = self.__builder.build(species)
		trainer = Trainer(
			model,
			loss_function=nn.CrossEntropyLoss(),
			optimizer=Adam(model.parameters(), lr=1e-3),
			callbacks=self.__trainer_callback
		)
		trainer.train(
			dataloader=self.__dataloader,
			epochs=self.__epochs,
			progress=True
		)

		return 1/trainer.validate(self.__test_dataloader)


class CNNOptimizer(Optimizer):

	@staticmethod
	def __generate_random_cnn_config(vocab_size: int) -> CNNConfig:
		num_layers = random.randint(3, 18)
		layers = []
		input_size = vocab_size
		for i in range(num_layers):
			kernel_floor, kernel_ceil = 3, min(7, input_size)
			if kernel_ceil <= kernel_floor:
				break
			kernel_size = random.randint(3, min(7, input_size))
			features = random.randint(32, 2024)
			padding = random.randint(0, min(3, input_size - kernel_size))
			pooling = random.randint(0, min(2, input_size - kernel_size - 2 * padding))
			layer = ConvLayer(kernel_size, features, padding, pooling)
			layers.append(layer)
			# Update input size for next layer
			input_size = (input_size - kernel_size + 2 * padding) //(pooling + 1)
		dropout = random.uniform(0, 0.5)
		config = CNNConfig(vocab_size, layers, dropout)
		return config

	def _generate_initial_generation(self) -> List[ModelConfig]:
		return [
			self.__generate_random_cnn_config(self._vocab_size)
			for _ in range(int(self._population_size))
		]


class TransformerOptimizer(Optimizer):

	def __init__(self, *args, block_size=1024, **kwargs):
		super().__init__(*args, **kwargs)
		self.__block_size = block_size

	def __generate_random_transformer_config(self, vocab_size: int) -> TransformerConfig:
		kernel_size = random.randint(1, 7)
		num_heads = random.randint(1, 16)
		emb_size = num_heads * random.randint(2, 32)
		block_size = self.__block_size
		ff_size = random.randint(64, 1024)
		config = TransformerConfig(vocab_size, kernel_size, emb_size, block_size, num_heads, ff_size)
		return config

	def _generate_initial_generation(self) -> List[Species]:
		return [
			self.__generate_random_transformer_config(self._vocab_size)
			for _ in range(int(self._population_size))
		]
