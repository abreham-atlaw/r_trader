import unittest
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

from core import Config
from core.utils.research.data.prepare import SimulationSimulator


class SimulationSimulatorTest(unittest.TestCase):

	def test_mock_data(self):

		SIZE = 1000

		dfs = [
			pd.DataFrame(
				data=np.concatenate(
					(
						j*np.expand_dims(np.arange(SIZE), axis=1),
						np.expand_dims(np.array([datetime.now().replace(microsecond=0, second=0, minute=0, hour=0) + timedelta(days=i) for i in range(SIZE)]), axis=1)
					),
					axis=1
				),
				columns=["c", "time"]
			)
			for j in [-1, 1]
		]

		simulator = SimulationSimulator(
			dfs=dfs,
			bounds=[1+p for p in [i*0.01 for i in range(-5, 5)]],
			seq_len=10,
			extra_len=2,
			batch_size=8,
			output_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data",
			granularity=3,
			ma_window=3,
			order_gran=True
		)
		simulator.start()

	def test_functionality(self):

		df = [
			pd.read_csv("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-10k.csv"),
			pd.read_csv("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-2k.csv")
		]

		simulator = SimulationSimulator(
			dfs=df,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			seq_len=1033,
			extra_len=124,
			batch_size=10,
			output_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data/ma_20",
			granularity=5,
			ma_window=20
		)

		simulator.start()
