import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime

from pymongo import MongoClient

from core.agent.utils.cache import Cache
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

	def __init__(
			self,
			sessions_repository: SessionsRepository,
			allowed_gpu_instances=2,
			allowed_cpu_instances=5,
			gpu_instance_usage=12
	):
		self.__sessions_repository = sessions_repository
		self.__allowed_gpu_instances = allowed_gpu_instances
		self.__allowed_cpu_instances = allowed_cpu_instances
		self.__gpu_instance_usage = gpu_instance_usage
		self.__cache = Cache()

	@abstractmethod
	def _get_resources(self, account: Account) -> Resources:
		pass

	@abstractmethod
	def _save_resources(self, resources: Resources):
		pass

	def get_resources(self, account: Account) -> Resources:
		resources = deepcopy(self.__cache.cached_or_execute(key=account.username, func=lambda: self._get_resources(account)))
		resources.get_resource(Resources.Devices.CPU).remaining_instances = self.__allowed_cpu_instances - len(
			self.__sessions_repository.filter(gpu=False, active=True, account=account))

		active_gpu_sessions = self.__sessions_repository.filter(gpu=True, active=True, account=account)
		for session in active_gpu_sessions:
			resources.get_resource(Resources.Devices.GPU).remaining_amount += max(
				0,
				self.__gpu_instance_usage - (datetime.now() - session.start_datetime).total_seconds()//3600
			)
		resources.get_resource(Resources.Devices.GPU).remaining_instances = self.__allowed_gpu_instances - len(
			self.__sessions_repository.filter(gpu=True, active=True, account=account))
		return resources

	def save_resources(self, resources: Resources):
		self.__cache.remove(resources.account.username)
		self._save_resources(resources)


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

	def _save_resources(self, resources: Resources):
		for resource in resources:
			self.__collection.update_one(
				{"device": resource.device, "account": resources.account.username},
				{"$set": resource.__dict__.copy()}
			)
