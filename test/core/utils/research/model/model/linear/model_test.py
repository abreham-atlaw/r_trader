import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.losses import WeightedMSELoss
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.wrapped import WrappedModel
from lib.utils.torch_utils.model_handler import ModelHandler
import matplotlib.pyplot as plt


class LinearTest(unittest.TestCase):

	@staticmethod
	def __plot_outputs(model: nn.Module, size: int):

		NP_DTYPE = np.float32

		def softmax(x):
			exp_x = np.exp(x - np.max(x))
			softmax_x = exp_x / np.sum(exp_x)
			return softmax_x

		def scale(x):
			x = softmax(x)
			x = x / np.max(x)
			return x

		X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/X/1727815242.844215.npy").astype(
			NP_DTYPE)[:size]
		y = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/y/1727815242.844215.npy").astype(
			NP_DTYPE)[:size]

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X)).detach().numpy()

		softmaxed = np.array([softmax(y_hat[i, :-1]) for i in range(y_hat.shape[0])])
		scaled = np.array([scale(y_hat[i, :-1]) for i in range(y_hat.shape[0])])

		for i in range(y_hat.shape[0]):
			plt.figure()
			plt.plot(y[i, :-1])
			plt.plot(softmaxed[i])

			# plt.figure()
			# plt.plot(X[i, :-124])

		y_hat_classes = np.argmax(y_hat, axis=1)
		y_classes = np.argmax(y, axis=1)

	def test_plot_outputs(self):

		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/results_3/abrehamalemu-rtrader-training-exp-0-linear-101-cum-0-it-4-tot.zip")

		self.__plot_outputs(model, 10)
		plt.show()

	def test_model_evolution(self):

		container_path = "/home/abrehamatlaw/Downloads/Compressed/results_1/out"

		models = [
			ModelHandler.load(os.path.join(container_path, f))
			for f in sorted(os.listdir(container_path))
			if f.endswith(".zip")
		]

		for model in models:
			self.__plot_outputs(model, 3)

		plt.show()

	def test_multiple_models_comparison(self):
		model_paths = [
			"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-77-cum-0-it-4-tot.zip",
			'/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-107-cum-0-it-4-tot.zip'
		]

		models = [
			ModelHandler.load(path)
			for path in model_paths
		]

		for model in models:
			self.__plot_outputs(model, 3)

		plt.show()

	def test_load_and_call(self):

		NP_DTYPE = np.float32

		model = ModelHandler.load(
			"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-104-cum-0-it-4-tot.zip"
		)

		X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/X/1727815242.844215.npy"
		).astype(
			NP_DTYPE
		)
		y = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/y/1727815242.844215.npy"
		).astype(
			NP_DTYPE
		)

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X)).detach().numpy()

		y_hat_classes = np.argmax(y_hat, axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertIsNotNone(y_hat_classes)
