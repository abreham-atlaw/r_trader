import typing

import numpy as np
from matplotlib import pyplot as plt

from core import Config
from core.utils.research.training.data.metric import MetricsContainer
from core.utils.research.training.data.repositories.metric_repository import MetricRepository, MongoDBMetricRepository
from lib.utils.logger import Logger


class TrainingPlotter:

	__SUBTITLES = ["State Classification", "Value Regression", "Total"]

	def __init__(self, threshold: float = None, fig_size: typing.Tuple[int, int] = (20, 10)):
		self.__threshold = threshold
		self.__fig_size = fig_size

	@staticmethod
	def __create_repo(notebook: str) -> MetricRepository:
		return MongoDBMetricRepository(
			Config.MONGODB_URL,
			notebook
		)

	def __apply_threshold(self, collection: np.ndarray) -> np.ndarray:
		collection = collection.copy()
		if self.__threshold is None:
			return collection
		mask = collection > self.__threshold
		if np.sum(mask) > 0:
			Logger.warning(f"Applying Threshold on {np.sum(mask)} metrics")
		collection[mask] = 0
		return collection

	def plot(self, notebook: str, show=True):
		Logger.info(f"Plotting {notebook}...")
		repo = self.__create_repo(notebook)

		metrics = MetricsContainer()
		for metric in repo.get_all():
			metrics.add_metric(metric)

		plt.figure(figsize=self.__fig_size)

		for i in range(3):
			train_losses, val_losses = [
				self.__apply_threshold(np.array([metric.value[i] for metric in metrics.filter_metrics(source=source)]))
				for source in [0, 1]
			]
			if train_losses.shape[0] == 0:
				Logger.error("Empty Metrics! Exiting...")
				return
			plt.subplot(2, 2, i+1)
			plt.title(f"{notebook}\n{self.__SUBTITLES[i]}\nMin: train: {min(train_losses)} val:{min(val_losses)}\n")
			plt.plot(train_losses)
			plt.plot(val_losses)
			plt.pause(0.1)

		if show:
			plt.show()

	def plot_multiple(self, notebooks: typing.List[str]):
		for notebook in notebooks:
			self.plot(notebook, show=False)
		plt.show()
