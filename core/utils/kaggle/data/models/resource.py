import typing
from abc import ABC, abstractmethod

from dataclasses import dataclass
from .account import Account


@dataclass
class Resource:
	device: int
	remaining_amount: typing.Optional[float]
	remaining_instances: int


class Resources:
	class Devices:

		CPU = 0
		GPU = 1
		TPU = 2

	def __init__(self, account: Account, resources: typing.Optional[typing.List[Resource]] = None):
		self.account = account
		self.__resources = resources
		if resources is None:
			self.__resources: typing.List[Resource] = []

	def __iter__(self):
		for resource in self.__resources:
			yield resource

	def __len__(self):
		return len(self.__resources)

	def __getitem__(self, idx):
		return self.__resources[idx]

	def get_resource(self, device) -> typing.Optional[Resource]:
		for resource in self.__resources:
			if resource.device == device:
				return resource
		return None
