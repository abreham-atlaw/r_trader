import unittest

from core import Config
from core.di import ResearchProvider, ServiceProvider
from core.utils.research.data.collect.runner_stats_branch_manager import RunnerStatsBranchManager


class RunnerStatsBranchManagerTest(unittest.TestCase):

	def setUp(self):
		self.client = ServiceProvider.provide_mongo_client()
		self.manager = RunnerStatsBranchManager()

	def test_sync_all(self):
		self.manager.sync_branches()

	def test_sync_branches(self):
		self.manager.sync_branches(
			branches=[
				Config.RunnerStatsBranches.ma_ews_dynamic_k_stm_it_23,
				Config.RunnerStatsBranches.ma_ews_dynamic_k_stm_it_23_tp_0,
			]
		)

	def test_import(self):

		filter_fn = lambda data: data["model_losses"][1] < 15

		db = self.client["runner_stats"]

		source_collection = db["runner_stats"]
		target_collection = db["runner_stats-branch-main"]

		all_ids = [doc["id"] for doc in target_collection.find()]

		valid_documents = list(filter(filter_fn, source_collection.find({})))

		for i, doc in enumerate(valid_documents):
			print(f"Processing {(i+1) * 100 / len(valid_documents):.2f}%...")
			if doc["id"] in all_ids:
				continue
			print(f"Importing {doc['id']}")
			target_collection.insert_one(doc)
