import unittest

from core import Config
from core.utils.research.utils.analysis.mlb_plot_rsa import ModelLossBranchesPlotRSAnalyzer


class ModelLossBranchesPlotRSATest(unittest.TestCase):

	def test_functionality(self):
		analyzer = ModelLossBranchesPlotRSAnalyzer(
			branches=[
				Config.RunnerStatsBranches.it_27_1
			],
			ml_branches=(
				Config.RunnerStatsLossesBranches.it_23_sw_11,
				Config.RunnerStatsLossesBranches.it_27
			)
		)
		analyzer.start()
