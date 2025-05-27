import math
import os.path
import random
import typing

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from core.utils.research.utils.model_analysis import f as F
from lib.utils.logger import Logger
from .model_analyzer import ModelAnalyzer
from .utils.plot_utils import PlotUtils


class LayerOutputModelAnalyzer(ModelAnalyzer):

	def __init__(
			self,
			*args,
			layers: typing.Dict[str, nn.Module] = None,
			export_path: str = "./",
			plot_samples: int = 10,
			plot_samples_random_state: int = 42,
			plot_cols: int = 2,
			plot_fig_size: typing.Tuple[int, int] = (10, 10),
			plot: bool = True,
			**kwargs
	):
		super(LayerOutputModelAnalyzer, self).__init__(*args, **kwargs)
		if layers is None:
			layers = {
				name: module
				for name, module in self._model.named_modules()
			}
		self.__layers = layers
		self.__export_path = os.path.abspath(export_path)
		self.__plot = plot
		self.__plot_samples_size = plot_samples
		self.__plot_samples_random_state = plot_samples_random_state
		self.__plot_fig_size = plot_fig_size
		self.__plot_cols = plot_cols

	def _process_output(self, output: torch.Tensor) -> torch.Tensor:
		return output

	def _generate_output_path(self, export_path: str, name: str) -> str:
		return os.path.join(export_path, f"layer_output-{name}.npy")

	def __get_plot_samples(self, X: torch.Tensor) -> torch.Tensor:
		idxs = torch.arange(X.shape[0]).tolist()
		r = random.Random(self.__plot_samples_random_state)
		r.shuffle(idxs)
		return X[idxs[:self.__plot_samples_size]]

	def __plot_samples(self, X: torch.Tensor, name: str):
		samples = self.__get_plot_samples(X)

		PlotUtils.plot(
			y=samples,
			title=name,
		)

	def __analyze_layer(self, model: nn.Module, X: torch.Tensor, layer: nn.Module, name: str):
		Logger.info(f"Analyzing layer \"{name}\"")
		output = F.get_layer_output(model, X, layer)
		output = self._process_output(output)

		export_path = self._generate_output_path(self.__export_path, name)
		Logger.info(f"Exporting layer output to {export_path}")
		np.save(export_path, output)

		if not self.__plot:
			return

		self.__plot_samples(output, name)

	def _analyze(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):

		for name, layer in self.__layers.items():
			try:
				self.__analyze_layer(model, X, layer, name)
			except ValueError as ex:
				Logger.error(f"Failed to analyze {name}: {ex}")

		if self.__plot:
			plt.show()

