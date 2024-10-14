import unittest

from pymongo import MongoClient
from torch.utils.data import DataLoader

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.collect.runner_stats_populater import RunnerStatsPopulater
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from core.utils.research.data.load.dataset import BaseDataset
from lib.utils.file_storage import PCloudClient, DropboxClient


class RunnerStatsPopulaterTest(unittest.TestCase):

	def test_functionality(self):

		repo = RunnerStatsRepository(
			MongoClient(Config.MONGODB_URL)
		)

		test_dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test"
			],
		)
		test_dataloader = DataLoader(test_dataset, batch_size=32)

		populater = RunnerStatsPopulater(
			repository=repo,
			dataloader=test_dataloader,
			in_filestorage=ServiceProvider.provide_file_storage("/"),
			in_path=Config.MODEL_IN_PATH,
			tmp_path="/tmp/",
		)

		populater.start()
