import json

import torch
from torch import nn
import matplotlib.pyplot as plt

from core.utils.research.utils.model_analysis import f as F
from core.utils.research.utils.model_analysis.model_analyzer.model_analyzer import ModelAnalyzer
from lib.utils.logger import Logger


class InputSaliencyModelAnalyzer(ModelAnalyzer):

	def __init__(
			self,
			*args,
			export_path: str = "saliency.json",
			plot: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__export_path = export_path
		self.__plot = plot

	def _analyze(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
		saliency = F.get_input_saliency(model, X)

		Logger.info(f"Exporting Saliency to {self.__export_path}")
		with open(self.__export_path, "w") as file:
			json.dump(saliency.tolist(), file)

		if self.__plot:
			plt.plot(saliency)
			plt.show()
