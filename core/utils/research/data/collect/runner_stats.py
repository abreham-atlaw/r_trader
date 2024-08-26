import typing

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunnerStats:
	id: str
	model_name: str
	session_timestamps: typing.List[datetime]
	profit: float = 0.0
	duration: float = 0.0
	model_losses: typing.Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	def add_profit(self, profit: float):
		self.profit += profit

	def add_duration(self, duration: float):
		self.duration += duration

	def add_session_timestamp(self, timestamp: datetime):
		self.session_timestamps.append(timestamp)
