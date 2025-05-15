import numpy as np
from xgboost import XGBRegressor

from core.utils.research.data.prepare.swg.training.models import SampleWeightGenerationModel
from core.utils.research.data.prepare.swg.training.models.layers import MinMaxNorm, Identity


class XGBoostSWGModel(SampleWeightGenerationModel):

	def __init__(
			self,
			norm: bool = False,
			**kwargs
	):
		self.model = XGBRegressor(**kwargs)
		self.norm = MinMaxNorm() if norm else Identity()

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict(X)

	def fit(self, X: np.ndarray, y: np.ndarray):
		X = self.norm(X)
		self.model.fit(X, y)
