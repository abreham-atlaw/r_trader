import typing

import matplotlib.pyplot as plt
import torch
from torch import nn

import os
import json

from lib.utils.logger import Logger
from .model_analyzer import ModelAnalyzer
from core.utils.research.utils.model_analysis.f import get_layer_saliency


class LayerSaliencyModelAnalyzer(ModelAnalyzer):

	def __init__(self, *args, layers: typing.Dict[str, nn.Module] = None, export_path: str = "./", plot: bool = True, **kwargs):
		super().__init__(*args, **kwargs)

		if layers is None:
			layers = {
				name: module
				for name, module in self._model.named_modules()
			}
		self.__layers = layers
		self.__export_path = os.path.abspath(export_path)
		self.__plot = plot

	def __analyze_layer(self, name: str, layer: nn.Module, model: nn.Module, X: torch.Tensor):
		Logger.info(f"Analyzing {name}...")
		saliency = get_layer_saliency(model, X, layer)

		export_path = os.path.join(self.__export_path, f"{name}.json")
		Logger.info(f"Exporting Layer Saliency to {export_path}")
		with open(export_path, 'w') as file:
			json.dump(saliency.tolist(), file)

		if self.__plot:
			plt.figure()
			plt.title(name)
			if saliency.ndim == 1:
				saliency = torch.unsqueeze(saliency, 0)

			for i in range(saliency.shape[0]):
				plt.plot(saliency[i], label=f"{i}")

			plt.legend()
			plt.pause(0.1)

	def _analyze(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):

		for i, (name, layer) in enumerate(self.__layers.items()):
			try:
				self.__analyze_layer(name, layer, model, X)
			except ValueError as ex:
				Logger.error(f"Failed to analyze {name}: {ex}")

		if self.__plot:
			plt.show()
