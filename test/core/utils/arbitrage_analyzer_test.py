import typing

import unittest

import numpy as np
import matplotlib.pyplot as plt

from core.utils.arbitrage_analyzer import ArbitrageAnalyzer


class ArbitrageAnalyzerTest(unittest.TestCase):

	SEQUENCE = (lambda x: np.sin(x) * np.sin(100*x))(np.linspace(0, np.pi*5, 1000))
	SEQUENCE1 = np.array([
		2.5, 2, 2.5, 3, 2.5, 2, 2.5, 3, 2.5, 2, 2.5, 3, 2.5, 2, 2.5, 3
	])
	CLOSE_ZONE_SIZE = 2.7
	BOUNCE_ZONE_SIZE = 0.9

	def setUp(self) -> None:
		# plt.plot(self.SEQUENCE)
		# plt.show()
		# plt.pause(0.01)
		pass

	def test_functionality(self):
		analyzer = ArbitrageAnalyzer()

		close_probability = analyzer.get_cross_probability(
			sequence=self.SEQUENCE1,
			zone_size=self.CLOSE_ZONE_SIZE,
			time_steps=6
		)
		self.assertTrue(isinstance(close_probability, float))

		bounce_probability = analyzer.get_bounce_probability(
			sequence=self.SEQUENCE1,
			close_zone_size=self.CLOSE_ZONE_SIZE,
			bounce_zone_size=self.BOUNCE_ZONE_SIZE,
			time_steps=7,
			bounces=3
		)
 		self.assertTrue(isinstance(bounce_probability, float))
