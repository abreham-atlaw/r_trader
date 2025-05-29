import typing

import matplotlib.pyplot as plt
import torch
from torch import nn

import os
import json

from lib.utils.logger import Logger
from .model_analyzer import ModelAnalyzer
from core.utils.research.utils.model_analysis.f import get_layer_saliency
from .utils.layer_utils import LayerUtils
from .utils.plot_utils import PlotUtils


class LayerSaliencyModelAnalyzer(ModelAnalyzer):

	def __init__(
			self,
			*args,
			layers: typing.Dict[str, nn.Module] = None,
			export_path: str = "./",
			plot: bool = True,
			skip_identity: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)

		if layers is None:
			layers = LayerUtils.get_layers(self._model, skip_identity=skip_identity)
		self.__layers = layers
		self.__export_path = os.path.abspath(export_path)
		self.__plot = plot

	def __analyze_layer(self, name: str, layer: nn.Module, model: nn.Module, X: torch.Tensor):
		Logger.info(f"Analyzing {name}...")
		saliency = get_layer_saliency(model, X, layer)

		export_path = os.path.join(self.__export_path, f"layer_saliency-{name}.json")
		Logger.info(f"Exporting Layer Saliency to {export_path}")
		with open(export_path, 'w') as file:
			json.dump(saliency.tolist(), file)

		if self.__plot:
			PlotUtils.plot(saliency, title=name)

	def _analyze(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):

		Logger.info(f"Analysing {len(self.__layers)} Layers...")

		for i, (name, layer) in enumerate(self.__layers.items()):
			try:
				self.__analyze_layer(name, layer, model, X)
			except (ValueError, RuntimeError) as ex:
				Logger.error(f"Failed to analyze {name}: {ex}")

		if self.__plot:
			plt.show()
