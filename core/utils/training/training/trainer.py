from typing import *

import numpy as np
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import psutil
import gc
import time

from core.utils.training.datapreparation.dataprocessor import DataProcessor
from core.utils.training.datapreparation.generators import WrapperGenerator
from .callbacks import Callback


class Trainer:

	def __init__(
			self,
			min_memory_percent: float = 30,
			test_size: float = 0.3,
			val_size: float = 0.3,
			batch_validation: bool = True
	):
		self.__min_memory_percent = min_memory_percent
		self.__test_size = test_size
		self.__val_size = val_size
		self.__batch_validation = batch_validation

		self.__set_variables(None, None, None, None, None)

		for size, name in [(test_size, "Test"), (val_size, "Val")]:
			if size < 0 or size > 1:
				raise ValueError(f"{name} size should be in range [0, 1]")

	def __monitor_memory(self):
		while (100 - psutil.virtual_memory().percent) < self.__min_memory_percent:
			print(f"[+]Awaiting memory release")
			gc.collect()
			time.sleep(5)

	@staticmethod
	def __split_data(secondary_size: float, indices: Optional[List[int]] = None, length: Optional[int] = None) -> Tuple[List[int], List[int]]:

		if indices is None:
			if length is None:
				raise ValueError("Either indices or length should be supplied")
			indices = list(range(length))

		return train_test_split(indices, test_size=secondary_size)

	def __split_train_val_test_data(self, processor: DataProcessor) -> Tuple[List[int], List[int], List[int]]:
		train_indices, test_indices = self.__split_data(self.__test_size, length=len(processor))
		train_indices, val_indices = self.__split_data(self.__val_size, indices=train_indices)

		indices = train_indices, val_indices, test_indices

		for ind in indices:
			ind.sort()

		return indices

	@staticmethod
	def __prepare_data(
			processor: DataProcessor,
			batch_idx: int,
			depth: int,
			start_depth: int = 0
	) -> Tuple[WrapperGenerator, WrapperGenerator]:
		core_generator, delta_generator = None, None

		for i in range(start_depth, depth):
			# print(f"[+]Processing Depth: {i + 1}/{depth}")
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

	def __evaluate_models(
			self,
			evaluation_indices: List[int]
	) -> Tuple[
			Union[
				Tuple[float, float],
				Tuple[float]
			],
			Union[
				Tuple[float, float],
				Tuple[float]
			]
	]:

		metrics = ([], [])

		for j in evaluation_indices:
			core_generator, delta_generator = self.__prepare_data(
				self.__processor,
				j,
				self.__depth
			)
			core_metrics, delta_metrics = [
				model.evaluate(generator, verbose=2)
				for model, generator in zip(self.__models, (core_generator, delta_generator))
			]
			if isinstance(core_metrics, float):
				core_metrics = (core_metrics,)
			if isinstance(delta_metrics, float):
				delta_metrics = (delta_metrics,)

			metrics[0].append(core_metrics)
			metrics[1].append(delta_metrics)

		return tuple([tuple(np.mean(metric, axis=0)) for metric in metrics])

	def __validate_models(
			self,
	):
		print(f"Validating Models")
		core_metrics, delta_metrics = self.__evaluate_models(
			self.__indices[1]
		)

		print(f"Core Metrics: {core_metrics}")
		print(f"Delta Metrics: {delta_metrics}")

	def __set_variables(
			self,
			indices,
			models,
			depth,
			callbacks,
			processor
	):
		self.__indices = indices
		self.__models = models
		self.__depth = depth
		self.__callbacks = callbacks
		self.__processor = processor

	def __clear_variables(self):
		self.__indices, self.__models, self.__depth, self.__callbacks, self.__processor  = None, None, None, None, None

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

		self.__set_variables(
			self.__split_train_val_test_data(processor),
			(core_model, delta_model),
			depth,
			callbacks,
			processor
		)

		for e in range(epochs):

			for callback in callbacks:
				callback.on_epoch_start(core_model, delta_model, e)

			print(f"Fitting Models")
			for i, bch_idx in enumerate(self.__indices[0][start_batch:]):

				print("\n\n", "-" * 100, "\n\n", sep="")
				print(f"[+]Processing\t\tEpoch: {e + 1}/{epochs}\t\tBatch:{i + 1}/{len(self.__indices[0][start_batch:])}")
				print(f"[+]Used Memory: {psutil.virtual_memory().percent}%")
				for callback in callbacks:
					callback.on_batch_start(core_model, delta_model, bch_idx)

				core_generator, delta_generator = self.__prepare_data(
					processor,
					bch_idx,
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
					callback.on_batch_end(core_model, delta_model, bch_idx)

				if self.__batch_validation:
					self.__validate_models()

			for callback in callbacks:
				callback.on_epoch_end(core_model, delta_model, e)

			self.__validate_models()

			start_batch, start_depth = 0, 0

		print("Testing Models")
		core_metrics, delta_metrics = self.__evaluate_models(self.__indices[2])
		print(f"Core Metrics: {core_metrics}")
		print(f"Delta Metrics: {delta_metrics}")

		self.__clear_variables()
