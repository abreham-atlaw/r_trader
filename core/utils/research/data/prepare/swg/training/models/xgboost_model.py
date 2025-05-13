import numpy as np
from xgboost import XGBRegressor

from core.utils.research.data.prepare.swg.training.models import SampleWeightGenerationModel


class XGBoostSWGModel(SampleWeightGenerationModel):

	def __init__(self, **kwargs):
		self.model = XGBRegressor(**kwargs)

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict(X)

	def fit(self, X: np.ndarray, y: np.ndarray):
		self.model.fit(X, y)

	def save(self, path: str):
		self.model.save_model(path)

	@classmethod
	def load(cls, path: str) -> 'XGBoostSWGModel':
		model = XGBoostSWGModel()
		model.model.load_model(path)
		return model
