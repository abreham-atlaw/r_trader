import typing
from copy import deepcopy

from core.Config import RunnerStatsBranches
from core.di import ResearchProvider
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository


class RunnerStatsBranchManager:

	def __init__(self, all_branches: typing.List[str] = None):
		if all_branches is None:
			all_branches = RunnerStatsBranches.all
		self.__all_branches = all_branches
		self.__repositories = {}

	def __get_repository(self, branch: str) -> RunnerStatsRepository:
		cached = self.__repositories.get(branch)
		if cached is None:
			self.__repositories[branch] = ResearchProvider.provide_runner_stats_repository(branch)
			return self.__get_repository(branch)
		return cached

	def __sync_branch(
			self,
			branch: str,
			synced_stats: typing.List[RunnerStats],
			branches: typing.List[str]
	) -> typing.List[RunnerStats]:

		synced_ids = [stat.id for stat in synced_stats]

		stats = self.__get_repository(branch=branch).retrieve_all()

		for target_branch in branches:
			if target_branch == branch:
				continue

			print(f"Syncing '{branch}' -> '{target_branch}'")

			target_stats = self.__get_repository(branch=target_branch).retrieve_all()
			target_ids = [stat.id for stat in target_stats]

			for i, stat in enumerate(stats):
				print(f"Processing {(i+1)*100/len(stats): .2f}%...")
				if stat.id in synced_ids:
					continue

				if stat.id in target_ids:
					continue

				new_stat = deepcopy(stat)
				new_stat.session_timestamps, new_stat.profits, new_stat.real_profits, new_stat.duration, new_stat.branch = ([], [], [], 0, target_branch)
				self.__get_repository(target_branch).store(new_stat)
				print(f"Added {new_stat.id} to {target_branch}")

		return stats

	def sync_branches(self, branches: typing.List[str] = None):

		if branches is None:
			branches = self.__all_branches

		synced_stats = []

		for source_stat in branches:

			new_syncs = self.__sync_branch(source_stat, synced_stats, branches)
			synced_stats.extend(new_syncs)
