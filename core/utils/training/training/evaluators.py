import typing
from abc import ABC, abstractmethod

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from core.utils.training.datapreparation import DataProcessor
from core.utils.training.datapreparation.generators import WrapperGenerator


class Evaluator(ABC):

	def __init__(self, depth: int, processor: typing.Optional[DataProcessor]=None, evaluation_indices=None):
		self.__processor = processor
		self.__indices = evaluation_indices
		self._depth = depth

	def _get_indices(self) -> typing.List[int]:
		if self.__indices is None:
			return list(range(len(self._get_processor())))
		return self.__indices

	def _get_processor(self):
		if self.__processor is None:
			raise ValueError("Data not set yet.")
		return self.__processor

	def set_processor(self, processor: DataProcessor):
		self.__processor = processor

	def set_indices(self, indices: typing.List[int]):
		self.__indices = indices

	def _get_data(self) -> typing.Tuple[WrapperGenerator, WrapperGenerator]:

		core_generator, delta_generator = None, None
		for i in range(self._depth):
			for idx in self._get_indices():
				depth_core_generator, depth_delta_generator = self._get_processor().get_data(idx, i)
				if core_generator is None:
					core_generator, delta_generator = depth_core_generator, depth_delta_generator
					continue
				core_generator.merge(depth_core_generator)
				delta_generator.merge(depth_delta_generator)
				depth_core_generator.destroy()
				depth_delta_generator.destroy()
				del depth_core_generator, depth_delta_generator

		return core_generator, delta_generator

	@abstractmethod
	def evaluate(self, core_model: Model, delta_model: Model) -> typing.Tuple[float, ...]:
		pass


class SingleModelEvaluator(Evaluator, ABC):

	@abstractmethod
	def _get_model(self, core_model: Model, delta_model: Model) -> Model:
		pass

	@abstractmethod
	def _get_single_model_data(self, core_generator: WrapperGenerator, delta_generator: WrapperGenerator) -> WrapperGenerator:
		pass

	def evaluate(self, core_model: Model, delta_model: Model) -> typing.Tuple[float, ...]:
		model = self._get_model(core_model, delta_model)
		return model.evaluate(self._get_single_model_data(*self._get_data()))


class CoreModelEvaluator(SingleModelEvaluator):

	def _get_single_model_data(self, core_generator: WrapperGenerator, delta_generator: WrapperGenerator) -> WrapperGenerator:
		return core_generator

	def _get_model(self, core_model: Model, delta_model: Model) -> Model:
		return core_model


class DeltaModelEvaluator(SingleModelEvaluator):

	def _get_single_model_data(self, core_generator: WrapperGenerator, delta_generator: WrapperGenerator) -> WrapperGenerator:
		return delta_generator

	def _get_model(self, core_model: Model, delta_model: Model) -> Model:
		return delta_model


class RegressionModelEvaluator(SingleModelEvaluator):

	def __init__(self, depth: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__depth = depth

	def __create_model(self, core_model: Model, delta_model: Model) -> Model:
		seq_len = core_model.input_shape[1]
		input_layer = layers.Input((seq_len,))

		sequence = input_layer
		for i in range(self.__depth):
			core_input = layers.Concatenate(axis=1)((sequence, i))
			probabilities = core_model.predict(core_input)

			delta_dec = delta_model.predict(layers.Concatenate(axis=1)((sequence, 0, i)))
			delta_inc = delta_model.predict(layers.Concatenate(axis=1)((sequence, 1, i)))

			delta = probabilities*delta_inc - (1-probabilities)*delta_dec

			sequence = layers.Concatenate(axis=1)((
				sequence,
				sequence[:, -1:] + delta
			))

		return sequence[:, -self.__depth:]

	def _get_single_model_data(self, core_generator: WrapperGenerator, delta_generator: WrapperGenerator) -> WrapperGenerator:
		pass

	def _get_model(self, core_model: Model, delta_model: Model) -> Model:
		return self.__create_model(core_model, delta_model)

