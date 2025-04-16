import json
import typing
from abc import ABC, abstractmethod
from datetime import datetime


class TimesRepository(ABC):

	@abstractmethod
	def retrieve_all(self) -> typing.List[datetime]:
		pass


class JsonTimesRepository(TimesRepository):

	def __init__(
			self,
			path: str,
			format="%Y-%m-%d %H:%M:%S+00:00"
	):
		with open(path, "r") as file:
			self.__times = map(
				lambda time_string: datetime.strptime(time_string, format),
				json.load(file)
			)

	def retrieve_all(self) -> typing.List[datetime]:
		return self.__times
