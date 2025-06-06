import typing
from abc import ABC, abstractmethod


class MCResourceManager(ABC):

	@abstractmethod
	def init_resource(self) -> typing.Any:
		pass

	@abstractmethod
	def has_resource(self, resource: typing.Any) -> bool:
		pass
