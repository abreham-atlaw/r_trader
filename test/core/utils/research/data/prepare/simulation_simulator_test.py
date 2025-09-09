import unittest
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

from core import Config
from core.utils.research.data.prepare import SimulationSimulator
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage, KalmanFilter


class SimulationSimulatorTest(unittest.TestCase):

	def test_mock_data(self):

		SIZE = 1000

		df = pd.DataFrame(
			data=np.concatenate(
				(
					np.expand_dims(np.arange(SIZE), axis=1),
					np.expand_dims(np.array([datetime.now() + timedelta(days=i) for i in range(SIZE)]), axis=1)
				),
				axis=1
			),
			columns=["c", "time"]
		)

		simulator = SimulationSimulator(
			df=df,
			bounds=[1+p for p in [i*0.01 for i in range(-5, 5)]],
			seq_len=10,
			extra_len=2,
			batch_size=8,
			output_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data",
			granularity=3,
			ma_window=3,
			order_gran=False
		)
		simulator.start()

	def test_functionality(self):

		df = pd.read_csv("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-10k.csv")

		simulator = SimulationSimulator(
			df=df,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			seq_len=1033,
			extra_len=124,
			batch_size=10,
			output_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/simulation_simulator_data/02",
			granularity=5,
			smoothing_algorithm=KalmanFilter(alpha=0.05, beta=0.01),
			order_gran=True
		)

		simulator.start()


