import typing

from core.utils.kaggle.data.models import Account, Resources
from core.utils.kaggle.data.repositories import ResourcesRepository, AccountsRepository


class ResourcesManager:

	def __init__(
			self,
			accounts_repository: AccountsRepository,
			resources_repository: ResourcesRepository,
	):
		self.__accounts_repository, self.__resources_repository = (
			accounts_repository, resources_repository)

	def __get_resources(self) -> typing.List[Resources]:
		return [
			self.__resources_repository.get_resources(account)
			for account in self.__accounts_repository.get_accounts()
		]

	def _sort_resources(self, resources: typing.List[Resources], device: int) -> typing.List[Resources]:

		key = lambda res: res.remaining_instances
		if device == Resources.Devices.GPU:
			key = lambda res: res.remaining_instances * res.remaining_amount

		return sorted(
			resources,
			key=lambda res: key(res.get_resource(device)),
			reverse=True
		)

	def __allocate_notebook(self, resources: Resources, device: int):
		resource = resources.get_resource(device)
		if resource.remaining_instances <= 0 or (resource.remaining_amount is not None and resource.remaining_amount <= 0):
			raise ResourceUnavailableException()
		resource.remaining_instances -= 1
		if resource.remaining_amount is not None:
			resource.remaining_amount -= 12
		self.__resources_repository.save_resources(resources)

	def allocate_notebook(self, use_gpu=True) -> Account:
		device = Resources.Devices.CPU
		if use_gpu:
			device = Resources.Devices.GPU
		accounts_resources = self.__get_resources()
		sorted_resources = self._sort_resources(accounts_resources, device)
		optimal_resources = sorted_resources[0]
		self.__allocate_notebook(optimal_resources, device)
		return optimal_resources.account

	def release_notebook(self, account: Account, use_gpu=True):
		resources = self.__resources_repository.get_resources(account)
		resource = resources.get_resource(Resources.Devices.CPU)
		if use_gpu:
			resource = resources.get_resource(Resources.Devices.GPU)
		resource.remaining_instances += 1
		self.__resources_repository.save_resources(resources)


class ResourceUnavailableException(Exception):
	pass
