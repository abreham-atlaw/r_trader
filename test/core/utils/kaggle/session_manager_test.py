import unittest

from kaggle.rest import ApiException

from core.di import init_di, ApplicationContainer
from core.utils.kaggle import SessionsManager
from core.utils.kaggle.data.models import Account, Resources
from core.utils.kaggle.data.repositories import SessionsRepository, AccountsRepository


class SessionManagerTest(unittest.TestCase):

	def setUp(self):
		init_di()

		self.accounts_repository: AccountsRepository = ApplicationContainer.kaggle.accounts_repository()
		self.manager: SessionsManager = ApplicationContainer.kaggle.sessions_manager()

	def test_functionality(self):
		account = (self.accounts_repository.get_by_username("abrehamatlaw0"))
		# account = Account(
		# 	username='yosephmezemer',
		# 	key='022b607f1ca94bc82cf68914eb6b0c4a'
		# )
		# account = Account(
		# 	username='bemnetatlaw',
		# 	key='0c9625e07a328c93a9c27fb1dda49f1a'
		# )
		# manager.sync_notebooks()
		self.manager.start_session(
			kernel="abrehamalemu/rtrader-training-exp-0-cnn-151-cum-0-it-4-tot-tpu",
			account=account,
			meta_data={
				"dataset_sources": [
					f"abrehamatlaw0/rtrader-datapreparer-simsim-cum-0-it-2-{i}"
					for i in range(4)
				]
			},
			device=Resources.Devices.GPU,
			sync_notebooks=False
		)

	def test_sync(self):
		self.manager.sync_notebooks()
