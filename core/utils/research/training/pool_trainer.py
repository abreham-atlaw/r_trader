import typing
from multiprocessing import Process

from threading import Thread

from .data.trainer_config import TrainConfig
from .trainer import Trainer


class TrainingProcess(Process):

	def __init__(self, create_config: typing.Callable):
		super().__init__()
		self.__create_config = create_config

	def run(self) -> None:
		config = self.__create_config()
		print("Running Trainer")
		trainer = Trainer(
			model=config.model,
			callbacks=config.callbacks,
		)
		config.compile(trainer)
		print("Starting Training")
		trainer.train(
			config.dataloader,
			val_dataloader=config.val_dataloader,
			epochs=config.epoch,
			progress=True,
			progress_interval=100,
			state=config.state
		)


class PoolTrainer:

	def train(
		self,
		configs: typing.List[typing.Callable]
	):

		processes = []
		for i, config in enumerate(configs):
			print(f"Starting process {i}")
			process = TrainingProcess(config)
			processes.append(process)
			process.start()

		for process in processes:
			process.join()
