import typing

import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.model.model.savable import SpinozaModule
from lib.rl.agent.dta import TorchModel
from lib.utils.torch_utils.model_handler import ModelHandler


class Lass(SmoothingAlgorithm):

	def __init__(self, model: typing.Union[SpinozaModule, str]):
		if isinstance(model, str):
			model = ModelHandler.load(model)
		self.__model = TorchModel(model)
		self.__window_size = model.input_size[1]

	def __apply(self, x: np.ndarray) -> np.ndarray:
		x = DataPrepUtils.stack(x, self.__window_size)
		y = self.__model.predict(x).flatten()
		return y

	def apply(self, x: np.ndarray) -> np.ndarray:
		return self.__apply(x)
