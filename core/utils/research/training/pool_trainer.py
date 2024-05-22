import typing

from threading import Thread

from .data.trainer_config import TrainConfig
from .trainer import Trainer


class TrainingProcess(Thread):

	def __init__(self, config: TrainConfig):
		super().__init__()
		self.__config = config

	def run(self) -> None:

		print("Running Trainer")
		trainer = Trainer(
			model=self.__config.model,
			callbacks=self.__config.callbacks,
		)
		self.__config.compile(trainer)
		print("Starting Training")
		trainer.train(
			self.__config.dataloader,
			val_dataloader=self.__config.val_dataloader,
			epochs=self.__config.epoch,
			progress=True,
			state=self.__config.state
		)


class PoolTrainer:

	def train(
		self,
		configs: typing.List[TrainConfig]
	):

		processes = []
		for i, config in enumerate(configs):
			print(f"Starting process {i}")
			process = TrainingProcess(config)
			processes.append(process)
			process.start()

		for process in processes:
			process.join()
