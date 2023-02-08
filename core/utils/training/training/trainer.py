from typing import *
from typing import Tuple

import numpy as np
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import psutil
import gc
import time
from dataclasses import dataclass

from core.utils.training.datapreparation.dataprocessor import DataProcessor
from core.utils.training.datapreparation.generators import WrapperGenerator
from .callbacks import Callback, CallbackException


class Trainer:

	@dataclass
	class Metric:

		source: int
		model: int
		epoch: int
		depth: int
		value: Tuple[float, ...]

	class MetricsContainer:

		def __init__(self):
			self.__metrics = []

		def add_metric(self, metric: 'Trainer.Metric'):
			self.__metrics.append(metric)

		def filter_metrics(self, source=None, epoch=None, depth=None, model=None) -> List['Trainer.Metric']:
			filtered_metrics = self.__metrics

			for attribute, value in zip(["source", "epoch", "depth", "model"], [source, epoch, depth, model]):
				if value is None:
					continue
				filtered_metrics = [metric for metric in filtered_metrics if metric.__dict__.get(attribute) == value]

			return filtered_metrics

		def get_metric(self, source=None, epoch=None, depth=None, model=None) -> Tuple[float,...]:
			metrics = self.filter_metrics(source=source, epoch=epoch, depth=depth, model=model)

			return tuple(np.mean([metric.value for metric in metrics], axis=0))

		def __iter__(self):
			for metric in self.__metrics:
				yield metric

		def __len__(self):
			return len(self.__metrics)

		def __getitem__(self, item):
			return self.__metrics[item]

	@dataclass
	class State:
		epoch: int
		epi: int
		batch: int
		depth: int

	def __init__(
			self,
			min_memory_percent: float = 30,
			test_size: float = 0.3,
			val_size: float = 0.3,
			batch_validation: bool = True,
			incremental: bool = False,
			increment_size: int = 1,
			random_state: Optional[int] = None
	):
		self.__min_memory_percent = min_memory_percent
		self.__test_size = test_size
		self.__val_size = val_size
		self.__batch_validation = batch_validation
		self.__incremental = incremental
		self.__increment_size = increment_size
		self.__state = None
		self.__random_state = random_state

		self.__set_variables(None, None, None, None, None, None)

		for size, name in [(test_size, "Test"), (val_size, "Val")]:
			if size < 0 or size > 1:
				raise ValueError(f"{name} size should be in range [0, 1]")

	def __monitor_memory(self):
		while (100 - psutil.virtual_memory().percent) < self.__min_memory_percent:
			print(f"[+]Awaiting memory release")
			gc.collect()
			time.sleep(5)

	def __split_data(self, secondary_size: float, indices: Optional[List[int]] = None, length: Optional[int] = None) -> Tuple[List[int], List[int]]:

		if indices is None:
			if length is None:
				raise ValueError("Either indices or length should be supplied")
			indices = list(range(length))

		return train_test_split(indices, test_size=secondary_size, random_state=self.__random_state)

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
	) -> Tuple[Metric, ...]:

		metrics = ([], [])

		for j in evaluation_indices:
			core_generator, delta_generator = self.__prepare_data(
				self.__processor,
				j,
				self.__depth
			)
			core_metrics, delta_metrics = [
				model.evaluate(generator, verbose=self.__verbose)
				for model, generator in zip(self.__models, (core_generator, delta_generator))
			]
			if isinstance(core_metrics, float):
				core_metrics = (core_metrics,)
			if isinstance(delta_metrics, float):
				delta_metrics = (delta_metrics,)

			metrics[0].append(core_metrics)
			metrics[1].append(delta_metrics)

		return tuple([
			Trainer.Metric(
				model=mi,
				source=2,
				epoch=0,
				depth=self.__depth,
				value=tuple(np.mean(metric, axis=0))
			)
			for mi, metric in enumerate(metrics)
		])

	def __validate_models(
			self,
			epoch
	) -> Tuple['Trainer.Metric', 'Trainer.Metric']:
		print(f"Validating Models")
		core_metrics, delta_metrics = self.__evaluate_models(
			self.__indices[1]
		)

		for metric in (core_metrics, delta_metrics):
			metric.source = 1
			metric.epoch = epoch

		print(f"Core Metrics: {core_metrics}")
		print(f"Delta Metrics: {delta_metrics}")
		return core_metrics, delta_metrics

	def __set_variables(
			self,
			indices,
			models,
			depth,
			callbacks,
			processor,
			verbose
	):
		self.__indices = indices
		self.__models = models
		self.__depth = depth
		self.__callbacks = callbacks
		self.__processor = processor
		self.__verbose = verbose

	def __clear_variables(self):
		self.__indices, self.__models, self.__depth, self.__callbacks, self.__processor, self.__verbose = None, None, None, None, None, None

	def fit(
			self,
			core_model: Model,
			delta_model: Model,
			processor: DataProcessor,
			depth: int,
			epochs: int = 1,
			epochs_per_inc=1,
			callbacks: List[Callback] = None,
			initial_state: 'Trainer.State' = None,
			verbose=2
	) -> 'Trainer.MetricsContainer':
		if callbacks is None:
			callbacks = []

		if initial_state is None:
			initial_state = Trainer.State(0, 0, 0, 1)

		if not self.__incremental:
			initial_state.depth, epochs_per_inc = depth, 1

		metrics = Trainer.MetricsContainer()

		state = Trainer.State(0, 0, 0, 1)

		for e in range(initial_state.epoch, epochs):
			state.epoch = e

			for callback in callbacks:
				try:
					callback.on_epoch_start(core_model, delta_model, state, metrics)
				except CallbackException:
					break

			for inc_depth in range(initial_state.depth, depth + 1, self.__increment_size):
				state.depth = inc_depth

				self.__set_variables(
					self.__split_train_val_test_data(processor),
					(core_model, delta_model),
					inc_depth,
					callbacks,
					processor,
					verbose
				)

				for epi in range(initial_state.epi, epochs_per_inc):
					state.epi = epi

					for callback in callbacks:
						try:
							callback.on_epoch_start(core_model, delta_model, state, metrics)
						except CallbackException:
							break

					print(f"Fitting Models")
					for i, bch_idx in enumerate(self.__indices[0][initial_state.batch:]):
						state.batch = i + initial_state.batch

						print("\n\n", "-" * 100, "\n\n", sep="")
						print(f"[+]Processing\t\tEpoch: {e + 1}/{epochs}\t\tInc Depth: {inc_depth}/{depth}\t\tInc Epoch: {epi + 1}/{epochs_per_inc}\t\tBatch:{i + 1}/{len(self.__indices[0][initial_state.batch:])}")
						print(f"[+]Used Memory: {psutil.virtual_memory().percent}%")
						for callback in callbacks:
							try:
								callback.on_batch_start(core_model, delta_model, state, metrics)
							except CallbackException:
								break

						core_generator, delta_generator = self.__prepare_data(
							processor,
							bch_idx,
							inc_depth,
						)
						print("[+]Fitting Core Model")
						core_metric = core_model.fit(core_generator, verbose=verbose)
						print("[+]Fitting Delta Model")
						delta_metric = delta_model.fit(delta_generator, verbose=verbose)

						for mi, metric in enumerate((core_metric, delta_metric)):
							metrics.add_metric(
								Trainer.Metric(
									source=0,
									model=mi,
									depth=inc_depth,
									epoch=e*epochs_per_inc + epi,
									value=tuple(np.array(list(metric.history.values())).flatten())
								)
							)

						core_generator.destroy()
						delta_generator.destroy()
						del core_generator, delta_generator
						gc.collect()

						for callback in callbacks:
							try:
								callback.on_batch_end(core_model, delta_model, state, metrics)
							except CallbackException:
								break

						if self.__batch_validation:
							self.__validate_models(e*epochs_per_inc + epi)

					for callback in callbacks:
						try:
							callback.on_epoch_end(core_model, delta_model, state, metrics)
						except CallbackException:
							break

					core_metric, delta_metric = self.__validate_models(e*epochs_per_inc + epi)
					for mi, metric in enumerate((core_metric, delta_metric)):
						metrics.add_metric(metric)

					initial_state.batch = 0
				initial_state.epi = 0
			if self.__incremental:
				initial_state.depth = 1

			for callback in callbacks:
				try:
					callback.on_epoch_end(core_model, delta_model, initial_state, metrics)
				except CallbackException:
					break

		print("Testing Models")
		core_metrics, delta_metrics = self.__evaluate_models(self.__indices[2])
		print(f"Core Metrics: {core_metrics}")
		print(f"Delta Metrics: {delta_metrics}")
		for mi, metric in enumerate((core_metrics, delta_metrics)):
			metrics.add_metric(metric)

		self.__clear_variables()

		return metrics
