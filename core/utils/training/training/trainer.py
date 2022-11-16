from typing import *

from tensorflow.keras.models import Model

import psutil
import gc
import time

from core.utils.training.datapreparation.dataprocessor import DataProcessor
from core.utils.training.datapreparation.generators import WrapperGenerator
from .callbacks import Callback


class Trainer:

	def __init__(self, min_memory_percent: float = 30):
		self.__min_memory_percent = min_memory_percent

	def __monitor_memory(self):
		while (100 - psutil.virtual_memory().percent) < self.__min_memory_percent:
			print(f"[+]Awaiting memory release")
			gc.collect()
			time.sleep(5)

	@staticmethod
	def __prepare_train_data(
			processor: DataProcessor,
			batch_idx: int,
			depth: int,
			start_depth: int = 0
	) -> Tuple[WrapperGenerator, WrapperGenerator]:
		core_generator, delta_generator = None, None

		for i in range(start_depth, depth):
			print(f"[+]Processing Depth: {i + 1}/{depth}")
			depth_core_generator, depth_delta_generator = processor.get_data(batch_idx, i)
			if core_generator is None:
				core_generator, delta_generator = depth_core_generator, depth_delta_generator
				continue
			core_generator.merge(depth_core_generator)
			delta_generator.merge(depth_delta_generator)
			depth_core_generator.destroy()
			depth_delta_generator.destroy()
			del depth_core_generator, depth_delta_generator
		core_generator.shuffle(), delta_generator.shuffle()
		return core_generator, delta_generator

	def fit(
			self,
			core_model: Model,
			delta_model: Model,
			processor: DataProcessor,
			depth: int,
			epochs: int = 1,
			callbacks: List[Callback] = None,
			start_batch=0,
			start_depth=0
	):
		if callbacks is None:
			callbacks = []

		for e in range(epochs):

			for callback in callbacks:
				callback.on_epoch_start(core_model, delta_model, e)

			for j in range(start_batch, len(processor)):
				print("\n\n", "-" * 100, "\n\n", sep="")
				print(f"[+]Processing\t\tEpoch: {e + 1}/{epochs}\t\tBatch:{j + 1}/{len(processor)}")
				print(f"[+]Used Memory: {psutil.virtual_memory().percent}%")
				for callback in callbacks:
					callback.on_batch_start(core_model, delta_model, j)

				core_generator, delta_generator = self.__prepare_train_data(
					processor,
					j,
					depth,
					start_depth=start_depth
				)
				print("[+]Fitting Core Model")
				core_model.fit(core_generator, verbose=2)
				print("[+]Fitting Delta Model")
				delta_model.fit(delta_generator, verbose=2)
				core_generator.destroy()
				delta_generator.destroy()
				del core_generator, delta_generator
				gc.collect()

				for callback in callbacks:
					callback.on_batch_end(core_model, delta_model, j)

				start_batch = 0

			for callback in callbacks:
				callback.on_batch_end(core_model, delta_model, e)

			start_depth = 0
