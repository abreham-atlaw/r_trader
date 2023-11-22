import unittest

from pymongo import MongoClient
from torch.utils.data import DataLoader

from core import Config
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
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/test"
			],
		)
		test_dataloader = DataLoader(test_dataset, batch_size=32)

		populater = RunnerStatsPopulater(
			repository=repo,
			dataloader=test_dataloader,
			in_filestorage=PCloudClient(Config.PCLOUD_API_TOKEN, "/Apps/RTrader/"),
			in_path="/Apps/RTrader/Models/Collected",
			out_filestorage=DropboxClient(
				Config.DROPBOX_API_TOKEN,
				"/Models/Collected/"
			),
			tmp_path="/tmp/",
			device="cpu"
		)

		populater.start()
