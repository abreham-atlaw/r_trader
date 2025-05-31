import unittest

import torch
import numpy as np

from core.utils.research.utils.model_migration.cnn_to_cnn2_migrator import CNNToCNN2Migrator
from lib.utils.torch_utils.model_handler import ModelHandler


class CNNToCNN2MigratorTest(unittest.TestCase):

	def setUp(self):
		self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-37-cum-0-it-35-tot.zip")
		self.migrator = CNNToCNN2Migrator()
		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/6/train/X/1740913843.59131.npy").astype(np.float32))

	def test_migrate(self):
		y = self.model(self.X)
		migrated = self.migrator.migrate(self.model)

		y_hat = migrated(self.X)

		self.assertTrue(torch.allclose(y, y_hat))
