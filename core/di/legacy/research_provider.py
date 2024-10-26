from core import Config


class ResearchProvider:

	@staticmethod
	def provide_profit_predictor() -> 'ProfitPredictor':
		from core.utils.research.data.collect.analysis import ProfitPredictor
		return ProfitPredictor()

	@staticmethod
	def provide_runner_stats_repository(branch=None) -> 'RunnerStatsRepository':
		from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
		from .service_provider import ServiceProvider

		if branch is None:
			branch = Config.RunnerStatsBranches.default

		return RunnerStatsRepository(
			client=ServiceProvider.provide_mongo_client(),
			profit_based_selection=False,
			branch=branch
		)
