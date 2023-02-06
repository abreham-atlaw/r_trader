import typing
from abc import ABC, abstractmethod

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from lib.dnn.layers import MovingAverage


class ModelWrapper(ABC):

	def __init__(self):
		pass

	@abstractmethod
	def wrap(self, model: Model) -> Model:
		pass


class CoreDeltaModelWrapper(ModelWrapper, ABC):

	def __init__(self):
		super().__init__()

	def _get_input_size(self, input_size: int) -> int:
		return input_size

	def _add_pre_filter(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer

	def _add_post_filter(self, out_layer: KerasTensor) -> KerasTensor:
		return out_layer

	def _get_prefilter_tensor(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer

	def _get_nonfilter_tensor(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer[:, :0]

	def _concat_post_non_filter_tensors(self, post_filter: KerasTensor, no_filter: KerasTensor) -> KerasTensor:
		return Concatenate(axis=1)((post_filter, no_filter))

	def wrap(self, model: Model) -> Model:
		inputs = Input((self._get_input_size(model.input_shape[1], )))
		post_filter = self._add_pre_filter(
			self._get_prefilter_tensor(
				inputs
			)
		)
		concat = self._concat_post_non_filter_tensors(
			post_filter,
			self._get_nonfilter_tensor(
				inputs
			)
		)
		out = self._add_post_filter(
			model(concat)
		)
		return Model(inputs=inputs, outputs=out)


class CoreModelWrapper(CoreDeltaModelWrapper):

	def _get_prefilter_tensor(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer[:, :-1]

	def _get_nonfilter_tensor(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer[:, -1:]


class DeltaModelWrapper(CoreDeltaModelWrapper):

	def _get_prefilter_tensor(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer[:, :-2]

	def _get_nonfilter_tensor(self, input_layer: KerasTensor) -> KerasTensor:
		return input_layer[:, -2:]


class MovingAverageModelWrapper(CoreDeltaModelWrapper):

	def __init__(self, average_window: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__average_window = average_window

	def _get_input_size(self, input_size: int) -> int:
		return input_size + self.__average_window - 1

	def _add_pre_filter(self, input_layer: KerasTensor) -> KerasTensor:
		return MovingAverage(self.__average_window)(input_layer)


class CoreMovingAverageWrapper(CoreModelWrapper, MovingAverageModelWrapper):
	pass


class DeltaMovingAverageWrapper(DeltaModelWrapper, MovingAverageModelWrapper):
	pass
