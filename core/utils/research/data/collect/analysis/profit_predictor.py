import typing

from core.utils.research.data.collect.analysis import RunnerStatsTrainer
from core.utils.research.data.collect.analysis.datapreparer import RunnerStatsDataPreparer
from core.utils.research.data.collect.analysis.models import XGBoostModel
from core.utils.research.data.collect.runner_stats import RunnerStats


class ProfitPredictor:

	def __init__(self, model=None, train=True):
		if model is None:
			model = self._init_model()
		self.__model = model
		self.__datapreparer = RunnerStatsDataPreparer(
			min_sessions=3
		)

	def _init_model(self):
		return XGBoostModel()

	def _train_model(self, model):
		trainer = RunnerStatsTrainer(
			data_preparer=self.__datapreparer
		)

		trainer.start(model)

	def predict(self, stat: RunnerStats) -> float:
		X, y = self.__datapreparer.prepare([stat])
		y_hat = self.__model.predict(X)
		return y_hat[0]
