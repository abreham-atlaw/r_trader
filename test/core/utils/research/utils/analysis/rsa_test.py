import unittest

from core import Config
from core.utils.research.utils.analysis.rs_filter import RSFilter
from core.utils.research.utils.analysis.rsa import RSAnalyzer


class RSAnalyzerTest(unittest.TestCase):

	def test_functionality(self):
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_23
		analyzer = RSAnalyzer(
			branches=[Config.RunnerStatsBranches.it_23_1, Config.RunnerStatsBranches.it_27_1],
			rs_filter=RSFilter(
				min_sessions=1
			),
			export_path="test.csv",
			sort_key=lambda stat: stat.model_losses[1]
		)
		analyzer.start()


