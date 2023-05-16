import typing
from abc import ABC, abstractmethod

from pymongo import MongoClient

from core.utils.kaggle.data.models.resource import Resource, Resources
from core.utils.kaggle.data.models.resource import Account
from . import SessionsRepository


class ResourcesRepository(ABC):

	@abstractmethod
	def get_resources(self, account: Account) -> Resources:
		pass

	@abstractmethod
	def save_resources(self, resources: Resources):
		pass


class SessionBasedResourcesRepository(ResourcesRepository, ABC):

	def __init__(self, sessions_repository: SessionsRepository, allowed_gpu_instances=2, allowed_cpu_instances=10):
		self.__sessions_repository = sessions_repository
		self.__allowed_gpu_instances = allowed_gpu_instances
		self.__allowed_cpu_instances = allowed_cpu_instances

	@abstractmethod
	def _get_resources(self, account: Account) -> Resources:
		pass

	def get_resources(self, account: Account) -> Resources:
		resources = self._get_resources(account)
		resources.get_resource(Resources.Devices.CPU).remaining_instances = self.__allowed_cpu_instances - len(
			self.__sessions_repository.filter(gpu=False, active=True, account=account))
		resources.get_resource(Resources.Devices.GPU).remaining_instances = self.__allowed_gpu_instances - len(
			self.__sessions_repository.filter(gpu=True, active=True, account=account))
		return resources


class MongoResourcesRepository(SessionBasedResourcesRepository):

	def __init__(self, session_repository: SessionsRepository,  mongo_client: MongoClient, db_name="kaggle", collection_name="resources"):
		super().__init__(session_repository)
		self.__collection = mongo_client[db_name][collection_name]

	def _get_resources(self, account: Account) -> Resources:
		resources_raw = self.__collection.find({"account": account.username})
		resources_list = []
		for resource_json in resources_raw:
			resource = Resource(*[None for _ in range(3)])
			resource.__dict__ = resource_json.copy()
			resource.__dict__.pop("account")
			resources_list.append(resource)
		return Resources(account, resources_list)

	def save_resources(self, resources: Resources):
		for resource in resources:
			self.__collection.update_one(
				{"device": resource.device, "account": resources.account.username},
				{"$set": resource.__dict__.copy()}
			)
