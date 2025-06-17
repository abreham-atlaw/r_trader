import numpy as np

from core.utils.research.data.prepare.swg.xswg.basic_xswg import BasicXSampleWeightGenerator


class MomentumXSampleWeightGenerator(BasicXSampleWeightGenerator):

	def __init__(self, *args, lookback: int = 5, **kwargs):
		super().__init__(*args, **kwargs)
		self.__lookback = lookback

	@staticmethod
	def __delta(X: np.ndarray) -> np.ndarray:
		return X[:, 1:] - X[:, :-1]

	def _generate(self, X: np.ndarray) -> np.ndarray:
		return (
			np.mean(np.abs(self.__delta(X[:, -self.__lookback:])), axis=1)
			/
			np.mean(np.abs(self.__delta(X)), axis=1)
		)
