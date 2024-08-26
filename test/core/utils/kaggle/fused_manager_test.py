import unittest

from core.di import init_di, ApplicationContainer
from core.utils.kaggle import SessionsManager, FusedManager
from core.utils.kaggle.data.repositories import SessionsRepository, AccountsRepository


class FusedManagerTest(unittest.TestCase):

	def test_functionality(self):
		init_di()

		manager: FusedManager = ApplicationContainer.kaggle.fused_manager()
		manager.start_session(
			kernel="inkoops/rtrader-maploss-runlive-sim-cnn-0",
			meta_data={
				"dataset_sources": ["abrehamatlaw0/rtrader-datapreparer-cum-0-it-1-v-0"],
			},
			sync_notebooks=False
		)
