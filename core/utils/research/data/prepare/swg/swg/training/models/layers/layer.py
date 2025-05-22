from abc import ABC

import numpy as np


class Layer(ABC):

	def _call(self, *args, **kwargs) -> np.ndarray:
		pass

	def __call__(self, *args, **kwargs) -> np.ndarray:
		return self._call(*args, **kwargs)
