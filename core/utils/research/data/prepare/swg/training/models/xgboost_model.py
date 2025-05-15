import numpy as np
from xgboost import XGBRegressor

from core.utils.research.data.prepare.swg.training.models import SampleWeightGenerationModel


class XGBoostSWGModel(SampleWeightGenerationModel):

	def __init__(
			self,
			norm: bool = False,
			**kwargs
	):
		super().__init__(norm=norm)
		self.model = XGBRegressor(**kwargs)

	def _predict(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict(X)

	def _fit(self, X: np.ndarray, y: np.ndarray):
		self.model.fit(X, y)
