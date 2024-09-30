import json
import unittest
from datetime import datetime

import matplotlib.pyplot as plt

from core.utils.research.data.prepare.bound_optimizer import BoundGenerator


class BoundOptimizerTest(unittest.TestCase):


	def test_functionality(self):

		N = 15
		EXPORT_PATH = f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/bounds/{datetime.now().timestamp()}.json"

		generator = BoundGenerator(
			start=0.9527526705012326,
			end=1.0336048383163439,
			csv_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-50k.csv",
			threshold=500,
		)

		bounds = generator.generate(N, plot=True)

		with open(EXPORT_PATH, "w") as f:
			json.dump(bounds, f)
			print(f"Exported to: {EXPORT_PATH}")

		self.assertEqual(len(bounds), N)
