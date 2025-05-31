import unittest

import numpy as np
from matplotlib import pyplot as plt

from core.utils.research.utils.model_analysis.model_analyzer.utils.plot_utils import PlotUtils


class PlotUtilsTest(unittest.TestCase):

	def test_plot(self):

		X = np.arange(0, 16*3000).reshape((16, 3000))**1.5

		PlotUtils.plot(
			y=X,
			title="My Plot",
			max_plots=5,
			mode=PlotUtils.Mode.IMAGE
		)

		plt.show()

