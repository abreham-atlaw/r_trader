import typing

from dataclasses import dataclass


@dataclass
class RunnerStats:
	id: str
	model_name: str
	profit: float = 0.0
	duration: float = 0.0
	model_losses: typing.Tuple[float, float] = (0.0, 0.0)
