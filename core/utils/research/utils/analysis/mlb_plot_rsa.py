import typing

import matplotlib.pyplot as plt
import numpy as np

from lib.utils.logger import Logger
from .rs_filter import RSFilter
from .rsa import RSAnalyzer
from ...data.collect.runner_stats import RunnerStats
from ...data.collect.runner_stats_populater import RunnerStatsPopulater


class ModelLossBranchesPlotRSAnalyzer(RSAnalyzer):

	__LOSS_NAMES = [
		str(loss)
		for loss in RunnerStatsPopulater.get_evaluation_loss_functions()
	]

	def __init__(
			self,
			branches: typing.List[str],
			ml_branches: typing.Tuple[str, str],
			export_path: str = "ml_branches.csv",
			extra_filter: typing.Optional[RSFilter] = None
	):
		assert len(ml_branches) == 2

		rs_filter = RSFilter(
			filter_fn=self.__custom_filter
		)
		if extra_filter is not None:
			rs_filter += extra_filter

		super().__init__(
			branches=branches,
			rs_filter=rs_filter,
			export_path=export_path,
		)
		self.__ml_branches = ml_branches

	def __custom_filter(self, stat: RunnerStats) -> bool:
		return True not in [
			0.0 in stat.get_model_losses(branch)
			for branch in self.__ml_branches
		]

	def __extract_model_loss(self, stats: typing.List[RunnerStats]) -> typing.Tuple[np.ndarray, np.ndarray]:
		losses_count = min([
			len(stat.get_model_losses(branch))
			for stat in stats
			for branch in self.__ml_branches
		])

		Logger.info(f"Using Losses count: {losses_count}")

		return tuple([
			np.array([
				stat.get_model_losses(branch)[:losses_count]
				for stat in stats
			])
			for branch in self.__ml_branches
		])

	def _handle_stats(self, stats: typing.List[RunnerStats]):
		super()._handle_stats(stats)

		X, y = self.__extract_model_loss(stats)

		for i in range(X.shape[1]):
			plt.figure()
			plt.title(self.__LOSS_NAMES[i])
			plt.grid(True)
			plt.scatter(X[:, i], y[:, i])

		plt.show()
