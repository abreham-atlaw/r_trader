import unittest
from datetime import datetime

import numpy as np

from core.di import ResearchProvider
from core.utils.research.data.collect.cleaning import TomCleaner
from lib.utils.torch_utils.model_handler import ModelHandler


class TomCleanerTest(unittest.TestCase):

	def setUp(self):
		self.repository = ResearchProvider.provide_runner_stats_repository()

		self.X = np.load("/home/abrehamatlaw/Downloads/1726500722.248052.npy").astype(np.float32)
		self.threshold = 0.5

		self.cleaner = TomCleaner(
			X=self.X,
			threshold=5,
			date_threshold=datetime(
				year=2024,
				month=10,
				day=20,
				hour=9,
				minute=30
			),
			in_path="/Apps/RTrader/maploss/models/linear/"
		)

	def test_get_threshold(self):
		model = ModelHandler.load('/home/abrehamatlaw/Downloads/Compressed/results_3/1728959384.707151.zip')
		print(self.cleaner.evaluate_model(model))

	def test_tom_cleaner(self):
		self.cleaner.start()
