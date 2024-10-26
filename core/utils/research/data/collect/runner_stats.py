import typing

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RunnerStats:

	id: str
	model_name: str
	session_timestamps: typing.List[datetime]
	profits: typing.List[float] = field(default_factory=lambda: [])
	duration: float = 0.0
	model_losses: typing.Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	temperature: float = 1.0
	real_profits: typing.List[float] = field(default_factory=lambda: [])

	@property
	def profit(self) -> float:
		return sum(self.profits)

	@property
	def real_profit(self) -> float:
		return sum(self.real_profits)

	def add_profit(self, profit: float):
		self.profits.append(profit)

	def add_real_profit(self, profit: float):
		self.real_profits.append(profit)

	def add_duration(self, duration: float):
		self.duration += duration

	def add_session_timestamp(self, timestamp: datetime):
		self.session_timestamps.append(timestamp)
