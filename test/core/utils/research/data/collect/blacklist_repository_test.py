import unittest

from core import Config
from core.di import ResearchProvider
from core.utils.research.data.collect.blacklist_repository import RSBlacklistRepository
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository


class RSBlacklistRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.branch = Config.RunnerStatsBranches.it_27_0
		self.rs_repo: RunnerStatsRepository = ResearchProvider.provide_runner_stats_repository(branch=self.branch)
		self.blacklist_repo: RSBlacklistRepository = ResearchProvider.provide_rs_blacklist_repository(rs_repo=self.rs_repo)

	def test_add(self):
		blacklisted = self.rs_repo.retrieve_all()[0].id
		self.blacklist_repo.add(blacklisted)
		self.assertEqual(self.blacklist_repo.is_blacklisted(blacklisted), True)

	def test_delete(self):
		blacklisted = self.rs_repo.retrieve_all()[0].id
		self.blacklist_repo.add(blacklisted)
		self.blacklist_repo.delete(blacklisted)
		self.assertEqual(self.blacklist_repo.is_blacklisted(blacklisted), False)
