import typing

import unittest

from core.utils.training.training.trainer import Trainer
from core.utils.training.training.continuoustrainer.repository import PCloudTrainerRepository

class PCloudTrainerRepositoryTest(unittest.TestCase):

	ID = "28"
	PATHS = ("path/to/0.h5", "path/to/1.h5")
	STATE = Trainer.State(
		epoch=4,
		epi=5,
		batch=50,
		depth=3
	)

	def test_functionality(self):
		repository = PCloudTrainerRepository("/Apps/RTrader")
		repository.update_checkpoint(
			self.ID,
			self.PATHS,
			self.STATE
		)

		paths, state = repository.get_checkpoint(self.ID)
		self.assertTupleEqual(self.PATHS, paths)
		self.assertEqual(self.STATE, state)
