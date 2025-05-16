import numpy as np

from core.utils.research.data.prepare.swg.training.models.layers import Layer


class Identity(Layer):

	def _call(self, x: np.ndarray) -> np.ndarray:
		return x
