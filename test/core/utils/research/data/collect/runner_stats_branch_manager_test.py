import unittest

from core.utils.research.data.collect.runner_stats_branch_manager import RunnerStatsBranchManager


class RunnerStatsBranchManagerTest(unittest.TestCase):

	def setUp(self):
		self.manager = RunnerStatsBranchManager()

	def test_sync_all(self):
		self.manager.sync_branches()
