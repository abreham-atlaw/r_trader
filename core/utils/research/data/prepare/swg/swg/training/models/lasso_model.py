import numpy as np
from sklearn.linear_model import Lasso

from .model import SampleWeightGenerationModel


class LassoSWGModel(SampleWeightGenerationModel):

	def __init__(self, *args, alpha: float = 1.0, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = Lasso(alpha=alpha)

	def _predict(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict(X)

	def _fit(self, X: np.ndarray, y: np.ndarray):
		self.model.fit(X, y)
