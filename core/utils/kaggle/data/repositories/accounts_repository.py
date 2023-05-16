import typing
from abc import ABC, abstractmethod

from pymongo import MongoClient

from core.utils.kaggle.data.models.account import Account


class AccountsRepository(ABC):

	@abstractmethod
	def get_accounts(self) -> typing.List[Account]:
		pass

	def get_by_username(self, username: str) -> typing.Optional[Account]:
		for account in self.get_accounts():
			if account.username == username:
				return account
		return None


class MongoAccountsRepository(AccountsRepository):

	def __init__(self, mongo_client: MongoClient, db_name="kaggle", collection_name="accounts"):
		super().__init__()
		self.__collection = mongo_client[db_name][collection_name]
		self.__cached = None

	def __get_cached(self) -> typing.Optional[typing.List[Account]]:
		return self.__cached

	def __get_from_db(self) -> typing.List[Account]:
		accounts_raw = self.__collection.find()
		accounts_list = []
		for account_json in accounts_raw:
			account = Account(*[None for i in range(2)])
			account.__dict__ = account_json.copy()
			accounts_list.append(account)
		return accounts_list

	def __cache(self, accounts: typing.List[Account]):
		self.__cached = accounts

	def get_accounts(self) -> typing.List[Account]:
		cached = self.__get_cached()
		if cached is not None:
			return cached
		from_db = self.__get_from_db()
		self.__cache(from_db)
		return from_db


class LocalAccountsRepository(AccountsRepository):

	def __init__(self, accounts: typing.List[Account]):
		self.__accounts = accounts

	def get_accounts(self) -> typing.List[Account]:
		return self.__accounts
