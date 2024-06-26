import typing

from tensorflow.keras.models import Model, load_model

import signal
import os
from dataclasses import dataclass

from lib.utils.file_storage import FileStorage
from core.utils.training.training.trainer import Trainer
from core.utils.training.datapreparation.dataprocessor import DataProcessor
from core.utils.training.training.callbacks import Callback
from .repository import TrainerRepository
from .callbacks import ContinuousTrainerCheckpointCallback, ContinuousTrainerCallback


class ContinuousTrainer(Trainer):

	@dataclass
	class StateMetricsTracker:
		state: typing.Optional[Trainer.State]
		metrics: typing.Optional[Trainer.MetricsContainer]

	class StateMetricsTrackerCallback(Callback):

		def __init__(self, tracker: 'ContinuousTrainer.StateMetricsTracker', *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.__tracker = tracker

		def on_epoch_start(self, core_model: Model, delta_model: Model, state: Trainer.State, metrics: Trainer.MetricsContainer):
			self.__tracker.state = state
			self.__tracker.metrics = metrics

	def __init__(self, repository: TrainerRepository, file_storage: FileStorage, *args, custom_objects=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.__repository = repository
		self.__file_storage = file_storage
		self.__custom_objects = custom_objects

	def __signal_handler(self, _, __):
		print("[-]Timeout")
		raise TimeoutException()

	def __prepare_callbacks(self, callbacks: typing.List[Callback], id):
		for callback in callbacks:
			if isinstance(callback, ContinuousTrainerCheckpointCallback):
				callback.init(id, self.__repository)

	@staticmethod
	def __download_model(url: str, out: str):
		os.system(f"wget --no-verbose {url} -O {out}")

	def __get_checkpoint(self, id: str) -> typing.Optional[typing.Tuple[typing.Tuple[Model, Model], Trainer.State]]:
		checkpoint = self.__repository.get_checkpoint(id)
		if checkpoint is None:
			return None

		models: typing.List[Model, Model] = []
		for path, type_ in zip(checkpoint[0], ContinuousTrainerCheckpointCallback.TYPES):
			filename = f"{type_}.h5"
			self.__file_storage.download(path, filename)
			models.append(load_model(filename, custom_objects=self.__custom_objects))
		models: typing.Tuple[Model, Model] = tuple(models)

		return models, checkpoint[1]

	def fit(
			self,
			id: str,
			core_model: Model,
			delta_model: Model,
			processor: DataProcessor,
			depth: int,
			epochs: int = 1,
			epochs_per_inc=1,
			callbacks: typing.List[Callback] = None,
			initial_state: 'Trainer.State' = None,
			verbose=2,
			timeout: typing.Optional[int] = None,
			core_training: bool = True,
			delta_training: bool = True
	) -> 'Trainer.MetricsContainer':

		self.__prepare_callbacks(callbacks, id)
		if timeout is not None:
			signal.signal(signal.SIGALRM, self.__signal_handler)
			signal.alarm(timeout)

		checkpoint = self.__get_checkpoint(id)
		if checkpoint is not None:
			(core_model, delta_model), initial_state = checkpoint
			processor.set_models(core_model, delta_model)

		tracker = ContinuousTrainer.StateMetricsTracker(None, None)

		callbacks += [ContinuousTrainer.StateMetricsTrackerCallback(tracker)]

		try:
			return super().fit(
				core_model,
				delta_model,
				processor,
				depth,
				epochs,
				epochs_per_inc,
				callbacks,
				initial_state,
				verbose,
				core_training,
				delta_training
			)

		except TimeoutException:
			for callback in callbacks:
				if isinstance(callback, ContinuousTrainerCallback):
					callback.on_timeout(core_model, delta_model, tracker.state, tracker.metrics)


class TimeoutException(Exception):
	pass
