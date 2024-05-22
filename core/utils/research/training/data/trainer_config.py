


from dataclasses import dataclass

import typing
from torch import nn
from torch.utils.data import DataLoader

from core.utils.research.training.callbacks import Callback
from core.utils.research.training.data.state import TrainingState


@dataclass
class TrainConfig:

	model: nn.Module
	compile: typing.Callable
	callbacks: typing.List[Callback]
	dataloader: DataLoader
	epoch: int = 1
	val_dataloader: typing.Optional[DataLoader] = None
	state: typing.Optional[TrainingState] = None
