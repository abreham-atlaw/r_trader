import typing
import random

from core.utils.kaggle.data.models import Account, Resources, Resource
from core.utils.kaggle.data.repositories import ResourcesRepository, AccountsRepository


class ResourcesManager:

	def __init__(
			self,
			accounts_repository: AccountsRepository,
			resources_repository: ResourcesRepository,
	):
		self.__accounts_repository, self.__resources_repository = (
			accounts_repository, resources_repository
		)
		self.__random_coefficient = 0.1

	def __get_resources(self) -> typing.List[Resources]:

		accounts = self.__accounts_repository.get_accounts()

		return [
			self.__resources_repository.get_resources(account)
			for account in accounts
		]

	def _sort_resources(self, resources: typing.List[Resources], device: int) -> typing.List[Resources]:

		def eval_cpu(res: Resource):
			return res.remaining_instances

		def eval_gpu(res: Resource):
			if res.remaining_instances <= 0 or res.remaining_amount <= 0:
				return 0
			return 1/((res.remaining_instances**1.5)*res.remaining_amount)

		def eval_tpu(res: Resource):
			return res.remaining_instances

		key = {
			Resources.Devices.CPU: eval_cpu,
			Resources.Devices.GPU: eval_gpu,
			Resources.Devices.TPU: eval_tpu
		}.get(device)

		return sorted(
			resources,
			key=lambda res: key(res.get_resource(device)) + self.__random_coefficient * random.random(),
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

	def release_notebook(self, account: Account, device=Resources.Devices.CPU):
		resources = self.__resources_repository.get_resources(account)
		resource = resources.get_resource(device)
		resource.remaining_instances += 1
		self.__resources_repository.save_resources(resources)

	def reset_resources(
			self,
			accounts: typing.List[Account] = None,
			gpu_amount=30,
			gpu_instances=2,
			cpu_instances=5,
			tpu_instances=1,
			tpu_amount=20
	):
		if accounts is None:
			accounts = self.__accounts_repository.get_accounts()

		for account in accounts:
			print(f"Resetting {account.username}...")
			resources = self.__resources_repository.get_resources(account)
			resources.get_resource(Resources.Devices.CPU).remaining_instances = cpu_instances
			resources.get_resource(Resources.Devices.GPU).remaining_instances = gpu_instances
			resources.get_resource(Resources.Devices.GPU).remaining_amount = gpu_amount
			resources.get_resource(Resources.Devices.TPU).remaining_instances = tpu_instances
			resources.get_resource(Resources.Devices.TPU).remaining_amount = tpu_amount
			self.__resources_repository.save_resources(resources)


class ResourceUnavailableException(Exception):
	pass
