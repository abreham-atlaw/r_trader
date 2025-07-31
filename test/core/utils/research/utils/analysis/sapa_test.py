import unittest

from matplotlib import pyplot as plt

from core.utils.research.data.prepare.smoothing_algorithm import KalmanFilter, MovingAverage
from core.utils.research.utils.analysis.sapa import SmoothingAlgorithmProfitabilityAnalyzer


class SmoothingAlgorithmProfitabilityAnalyzerTest(unittest.TestCase):

	def setUp(self):
		self.analyzer = SmoothingAlgorithmProfitabilityAnalyzer(
			df_path="/home/abrehamatlaw/Downloads/1750256680.835372.csv",
			view_size=120,
			tp_threshold=5,
			action_lag_size=(18, 20),
			plot_show=False,
			samples=4,
			plot=False
		)

		self.sas = [
			MovingAverage(64),
			MovingAverage(32),
			KalmanFilter(0.03, 0.001)
		]

	def test_analyze(self):
		for sa in self.sas:
			self.analyzer.analyze(sa)
		plt.show()
