import unittest

from core.di import init_di, ApplicationContainer
from core.utils.kaggle import SessionsManager, FusedManager
from core.utils.kaggle.data.models import Resources
from core.utils.kaggle.data.repositories import SessionsRepository, AccountsRepository


class FusedManagerTest(unittest.TestCase):

	def test_functionality(self):
		init_di()

		manager: FusedManager = ApplicationContainer.kaggle.fused_manager()
		manager.start_session(
			kernel="abrehamalemu/rtrader-training-exp-0-cnn-151-cum-0-it-4-tot",
			meta_data={
				"dataset_sources": [
					f"abrehamatlaw0/rtrader-datapreparer-simsim-cum-0-it-2-{i}"
					for i in range(4)
				]
			},
			device=Resources.Devices.GPU,
			sync_notebooks=False
		)
