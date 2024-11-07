import typing
from abc import ABC, abstractmethod

from pymongo import MongoClient

from core.agent.utils.cache import Cache
from core.utils.kaggle.data.models import Session, Account
from .accounts_repository import AccountsRepository


class SessionsRepository(ABC):

	@abstractmethod
	def add_session(self, session: Session):
		pass

	@abstractmethod
	def get_all(self) -> typing.List[Session]:
		pass

	@abstractmethod
	def finish_session(self, session: Session):
		pass

	@staticmethod
	def __fits(session: Session, filters: typing.Dict[str, typing.Any]) -> bool:

		for key, value in filters.items():
			if session.__dict__.get(key) != value:
				return False
		return True

	def filter(
			self,
			account: typing.Optional[Account] = None,
			active: typing.Optional[bool] = None,
			kernel: typing.Optional[str] = None,
			device: typing.Optional[int] = None
	) -> typing.List[Session]:
		sessions = self.get_all()

		filters = {
			key: value
			for key, value in zip(["account", "active", "device", "kernel"], [account, active, device, kernel])
			if value is not None
		}

		return [
			session
			for session in sessions
			if self.__fits(session, filters)
		]


class MongoSessionsRepository(SessionsRepository):

	def __init__(self, accounts_repository: AccountsRepository, mongo_client: MongoClient, db_name="kaggle", collection_name="sessions"):
		super().__init__()
		self.__accounts_repository = accounts_repository
		self.__collections = mongo_client[db_name][collection_name]
		self.__cache = Cache()

	def add_session(self, session: Session):
		session_json = session.__dict__.copy()
		session_json["account"] = session.account.username
		self.__collections.insert_one(session_json)

	def get_all(self) -> typing.List[Session]:
		sessions_raw = self.__collections.find()
		sessions_list = []
		for session_json in sessions_raw:
			session = Session(*[None for i in range(5)])
			session.__dict__ = {key: value for key, value in session_json.copy().items() if key in session.__dict__.keys()}
			session.account = self.__accounts_repository.get_by_username(session_json["account"])
			sessions_list.append(session)
		return sessions_list

	def finish_session(self, session: Session):
		session_json = session.__dict__.copy()
		session_json["account"] = session.account.username
		self.__collections.update_one(
			session_json,
			{"$set": {"active": False}}
		)
