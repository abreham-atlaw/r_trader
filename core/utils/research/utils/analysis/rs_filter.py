import typing
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RSFilter:

	model_key: str = None

	max_model_losses: typing.Tuple[float, float] = None
	min_model_losses: typing.Tuple[float, float] = None
	
	min_profit: float = None
	max_profit: float = None

	min_sessions: int = None

	max_temperature: float = None
	min_temperature: float = None

	evaluation_complete: bool = False

	def join(self, other: 'RSFilter') -> 'RSFilter':
		return RSFilter(
			model_key=self.model_key or other.model_key,
			max_model_losses=self.max_model_losses or other.max_model_losses,
			min_model_losses=self.min_model_losses or other.min_model_losses,
			min_profit=self.min_profit or other.min_profit,
			max_profit=self.max_profit or other.max_profit,
			min_sessions=self.min_sessions or other.min_sessions,
			max_temperature=self.max_temperature or other.max_temperature,
			min_temperature=self.min_temperature or other.min_temperature,
			evaluation_complete=self.evaluation_complete or other.evaluation_complete
		)

	def __add__(self, other):
		return self.join(other)