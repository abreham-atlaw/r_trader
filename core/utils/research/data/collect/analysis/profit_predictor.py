import typing

from .trainer import RunnerStatsTrainer
from .datapreparer import RunnerStatsDataPreparer
from core.utils.research.data.collect.analysis.models import XGBoostModel, RidgeModel, DecisionTreeModel, SVRModel, \
	KNNModel
from core.utils.research.data.collect.runner_stats import RunnerStats


class ProfitPredictor:

	def __init__(self, model=None):
		if model is None:
			model = self._init_model()
		self.__model = model
		self.__datapreparer_singleton = None
		self.__trained = False

	@property
	def __datapreparer(self):
		if self.__datapreparer_singleton is None:
			self.__datapreparer_singleton = RunnerStatsDataPreparer(
				min_sessions=3,
				# columns=[0],
			)
		return self.__datapreparer_singleton

	def _init_model(self):
		return KNNModel(k=3)

	def _train_model(self, model):
		trainer = RunnerStatsTrainer(
			data_preparer=self.__datapreparer
		)

		trainer.start(model)
		self.__trained = True

	def predict(self, stat: RunnerStats) -> float:
		if not self.__trained:
			self._train_model(self.__model)
		X, y = self.__datapreparer.prepare([stat])
		y_hat = self.__model.predict(X)
		return y_hat[0]
