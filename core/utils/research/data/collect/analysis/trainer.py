import typing

from core.utils.research.data.collect.analysis.datapreparer import RunnerStatsDataPreparer
from core.utils.research.data.collect.analysis.losses import Loss
from core.utils.research.data.collect.analysis.losses.mse import MeanSquaredError
from core.utils.research.data.collect.analysis.models import Model


class RunnerStatsTrainer:

	def __init__(self, loss_fn: Loss = None, data_preparer: RunnerStatsDataPreparer = None):
		if data_preparer is None:
			data_preparer = RunnerStatsDataPreparer()
		self.__data_preparer = data_preparer
		self.X, self.y = self.__data_preparer.prepare()

		if loss_fn is None:
			loss_fn = MeanSquaredError()
		self.__loss_fn = loss_fn

	def __evaluate(self, model: Model):
		y_hat = model.predict(self.X)
		return self.__loss_fn.evaluate(y_hat, self.y)

	def start(self, model: Model, epochs=1):
		for i in range(epochs):
			print(f"Epoch {i}")
			model.fit(self.X, self.y)
		return self.__evaluate(model)
