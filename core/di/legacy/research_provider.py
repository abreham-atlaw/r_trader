import typing

from .service_provider import ServiceProvider
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
			branch=branch,
		)

	@staticmethod
	def provide_times_repository() -> 'TimesRepository':
		from core.utils.research.data.collect.sim_setup.times_repository import JsonTimesRepository
		return JsonTimesRepository(Config.OANDA_SIM_TIMES_PATH)

	@staticmethod
	def provide_rs_setup_manager() -> 'RSSetupManager':

		from core.utils.research.data.collect.sim_setup.rs_setup_manager import RSSetupManager
		return RSSetupManager(
			fs=ServiceProvider.provide_file_storage(Config.OANDA_SIM_MODEL_IN_PATH),
			rs_repo=ResearchProvider.provide_runner_stats_repository(),
			times_repo=ResearchProvider.provide_times_repository()
		)

	@staticmethod
	def provide_ims_repository(model_name: str, label: str) -> 'IMSRepository':
		from core.utils.research.training.data.repositories.ims_repository import IMSRepository
		from core.di import ServiceProvider

		return IMSRepository(
			model_name=model_name,
			label=label,
			sync_size=Config.IMS_SYNC_SIZE,
			fs=ServiceProvider.provide_file_storage(Config.IMS_REMOTE_PATH),
			tmp_path=Config.IMS_TEMP_PATH
		)

	@staticmethod
	def provide_default_trackers(model_name: str) -> typing.List['TorchTracker']:
		return []
		# from core.utils.research.training.trackers.stats_tracker import DynamicStatsTracker
		# from core.utils.research.training.trackers.stats_tracker import Keys, WeightsStatsTracker, GradientsStatsTracker
		#
		# return [
		# 	DynamicStatsTracker(
		# 		model_name=model_name,
		# 		label=key
		# 	)
		# 	for key in Keys.ALL
		# ] + [
		# 	WeightsStatsTracker(
		# 		model_name=model_name
		# 	),
		# 	GradientsStatsTracker(
		# 		model_name=model_name
		# 	)
		# ]
		#
