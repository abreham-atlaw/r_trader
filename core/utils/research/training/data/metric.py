from dataclasses import dataclass

import typing

import numpy as np


@dataclass
class Metric:

	source: int
	model: int
	epoch: int
	depth: int
	value: typing.Tuple[float, ...]
	

class MetricsContainer:

	def __init__(self):
		self.__metrics = []

	def add_metric(self, metric: Metric):
		self.__metrics.append(metric)

	def filter_metrics(self, source=None, epoch=None, depth=None, model=None) -> typing.List[Metric]:
		filtered_metrics = self.__metrics

		for attribute, value in zip(["source", "epoch", "depth", "model"], [source, epoch, depth, model]):
			if value is None:
				continue
			filtered_metrics = [metric for metric in filtered_metrics if metric.__dict__.get(attribute) == value]

		return filtered_metrics

	def get_metric(self, source=None, epoch=None, depth=None, model=None) -> typing.Tuple[float, ...]:
		metrics = self.filter_metrics(source=source, epoch=epoch, depth=depth, model=model)

		return tuple(np.mean([metric.value for metric in metrics], axis=0))

	def __iter__(self):
		for metric in self.__metrics:
			yield metric

	def __len__(self):
		return len(self.__metrics)

	def __getitem__(self, item):
		return self.__metrics[item]
