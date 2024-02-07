from dataclasses import dataclass


@dataclass
class TrainingState:
	id: str
	epoch: int
	batch: int
