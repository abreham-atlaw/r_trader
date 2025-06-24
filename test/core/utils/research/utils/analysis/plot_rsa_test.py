import unittest

from core import Config
from core.utils.research.utils.analysis.plot_rsa import PlotRSAnalyzer


class PlotRSATest(unittest.TestCase):

	def test_functionality(self):
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_23
		analyzer = PlotRSAnalyzer(
			branches=[Config.RunnerStatsBranches.it_23_1, Config.RunnerStatsBranches.it_27_1],
			sessions_len=5
		)
		analyzer.start()
