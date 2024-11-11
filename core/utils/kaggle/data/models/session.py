import typing

from dataclasses import dataclass
from datetime import datetime

from . import Account


@dataclass
class Session:

	account: Account
	kernel: str
	active: bool
	device: int
	start_datetime: datetime = datetime.now()
