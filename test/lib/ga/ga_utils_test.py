import os.path
import random
import unittest

from core.utils.ganno.torch.concurrency.data.serializers import LinearConfigSerializer

from core import Config
from core.di import ServiceProvider
from core.utils.ganno.torch.nnconfig import LinearConfig
from lib.ga import GAUtils
from lib.ga.callbacks import StoreCheckpointCallback, CheckpointCallback


class GAUtilsTest(unittest.TestCase):

	def test_load(self):

		SAVE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/ga/population/population.ga"

		population = [
			LinearConfig(
				layers=[random.randint(0, 10) for _ in range(random.randint(0, 10))],
				dropout=0.5,
				vocab_size=5,
				norm=False
			)
			for _ in range(10)
		]

		callback = CheckpointCallback(
			species_serializer=LinearConfigSerializer(),
			save_path=SAVE_PATH
		)

		callback.on_epoch_end(population)

		loaded_population = GAUtils.load_population(
			SAVE_PATH,
			serializer=LinearConfigSerializer()
		)

		self.assertIsNotNone(loaded_population)
		self.assertTrue(isinstance(loaded_population, list))
		self.assertTrue(len(loaded_population) > 0)

		self.assertTrue(len(loaded_population) == len(population))
		self.assertEqual(loaded_population, population)

	def test_load_from_fs(self):
		loaded_population = GAUtils.load_from_fs(
			fs=ServiceProvider.provide_file_storage(path=Config.POPULATION_UPLOAD_PATH),
			# path=os.path.join(Config.POPULATION_UPLOAD_PATH, os.path.basename(Config.POPULATION_SAVE_PATH)),
			path=os.path.basename(Config.POPULATION_SAVE_PATH),
			serializer=LinearConfigSerializer()
		)

		self.assertIsNotNone(loaded_population)
		self.assertTrue(isinstance(loaded_population, list))
		self.assertTrue(len(loaded_population) > 0)