import os
import unittest

from pymongo import MongoClient
from torch.utils.data import DataLoader

from core import Config
from core.di import ServiceProvider, ResearchProvider
from core.utils.research.data.collect.runner_stats_populater import RunnerStatsPopulater
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from core.utils.research.data.load.dataset import BaseDataset
from lib.utils.file_storage import PCloudClient, DropboxClient


class RunnerStatsPopulaterTest(unittest.TestCase):

	def test_functionality(self):

		repo = ResearchProvider.provide_runner_stats_repository(Config.RunnerStatsBranches.main)

		test_dataset = BaseDataset(
			[
				os.path.join(Config.BASE_DIR, "temp/Data/prepared/7/train")
			],
		)
		test_dataloader = DataLoader(test_dataset, batch_size=32)

		IN_PATH = "/Apps/RTrader/maploss/it-42/"

		populater = RunnerStatsPopulater(
			repository=repo,
			dataloader=test_dataloader,
			in_filestorage=ServiceProvider.provide_file_storage(path="/"),
			in_path=IN_PATH,
			raise_exception=False,
			horizon_mode=True,
			horizon_h=0.3,
			horizon_bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)

		populater.start()
