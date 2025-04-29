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
