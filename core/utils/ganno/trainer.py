import typing
from abc import ABC, abstractmethod

from tensorflow.keras.models import Model

from core.utils.training.datapreparation import DataProcessor
from core.utils.training.training import Trainer


class GannoTrainer(Trainer, ABC):

	def __init__(self, depth: int, *args, epochs: int = 1, **kwargs):
		super().__init__(*args, **kwargs)
		self.__depth = depth
		self.__epochs = epochs

	@abstractmethod
	def _init_processor(self, core_model, delta_model) -> DataProcessor:
		pass

	def fit(
			self,
			core_model: Model,
			delta_model: Model,
	) -> 'Trainer.MetricsContainer':
		processor = self._init_processor(core_model, delta_model)
		return super().fit(
			core_model,
			delta_model,
			processor,
			self.__depth,
			self.__epochs
		)

