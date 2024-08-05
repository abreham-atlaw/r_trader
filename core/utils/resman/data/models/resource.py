import typing
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Resource:

	id: str
	lock_datetime: typing.Optional[datetime]

	@property
	def locked(self) -> bool:
		return self.lock_datetime is not None
