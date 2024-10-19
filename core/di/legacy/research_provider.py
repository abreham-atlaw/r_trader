

class ResearchProvider:

	@staticmethod
	def provide_profit_predictor() -> 'ProfitPredictor':
		from core.utils.research.data.collect.analysis import ProfitPredictor
		return ProfitPredictor()

	@staticmethod
	def provide_runner_stats_repository() -> 'RunnerStatsRepository':
		from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
		from .service_provider import ServiceProvider
		return RunnerStatsRepository(
			client=ServiceProvider.provide_mongo_client(),
			profit_based_selection=True
		)
