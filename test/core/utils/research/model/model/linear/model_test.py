import unittest

import numpy as np
import torch
from torch import nn

import os

from core import Config
from core.utils.research.eval.live_prediction.live_predictor import LivePredictor
from core.utils.research.model.model.utils import WrappedModel, TransitionOnlyModel, TemperatureScalingModel
from lib.rl.agent.dta import TorchModel, Model
from lib.utils.math import softmax, moving_average
from lib.utils.torch_utils.model_handler import ModelHandler
import matplotlib.pyplot as plt

from temp import stats


class LinearTest(unittest.TestCase):

	def setUp(self):
		NP_DTYPE = np.float32
		self.model = TemperatureScalingModel(
			ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-6-it-42-tot.zip"),
			temperature=1.0
		)
		self.model.eval()
		self.X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/X/1743180011.758194.npy").astype(
			NP_DTYPE
		)
		self.y = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/y/1743180011.758194.npy").astype(
			NP_DTYPE)
		self.tom_model = TransitionOnlyModel(
			model=self.model,
			extra_len=Config.AGENT_MODEL_EXTRA_LEN
		)
		self.wrapped_model = WrappedModel(
			self.tom_model,
			seq_len=Config.MARKET_STATE_MEMORY,
			window_size=Config.AGENT_MA_WINDOW_SIZE
		)
		self.torch_model = TorchModel(
			self.wrapped_model
		)
		self.bounds = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND + [Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND[-1] + Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON]
		self.bounds_centered = np.array([np.mean(self.bounds[i: i+2]) for i in range(len(self.bounds)-1)], dtype=np.float32)
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

		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-98-cum-0-it-4-tot.zip")

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

		with torch.no_grad():
			y_hat: torch.Tensor = self.model(torch.from_numpy(self.X)).detach().numpy()

		y_hat_classes = np.argmax(y_hat, axis=1)
		y_classes = np.argmax(self.y, axis=1)

		self.assertIsNotNone(y_hat_classes)

	def __call_from_state(self, model: nn.Module, dump_path: str, Ks=None, title: str = "", use_softmax=True, path=None):

		def get_node(root, path):
			path = path.copy()
			node = root
			while len(path) > 0:
				node = node.get_children()[path.pop(0)]
			return node

		if Ks is None:
			Ks = [len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) - 1]

		node, repo = stats.load_node_repo(
			dump_path
		)
		if path is not None:
			node = get_node(node, path)
		state = repo.retrieve(node.id)
		self.__plot_output_from_sequence(
			model,
			state.market_state.get_state_of("AUD", "USD"),
			Ks=Ks,
			title=title,
			use_softmax=use_softmax,
		)

	def __plot_output_from_sequence(self, model: nn.Module, sequence: np.ndarray, Ks=None, title: str = "", use_softmax=True):
		
		if Ks is None:
			Ks = [len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) - 1]

		X = np.expand_dims(
			np.concatenate([
				sequence,
				np.zeros(Config.AGENT_MODEL_EXTRA_LEN)
			]),
			axis=0
		).astype(np.float32)

		with torch.no_grad():
			if isinstance(model, Model):
				y_hat: np.ndarray = model.predict(X)
			else:
				y_hat: np.ndarray = model(
					torch.from_numpy(X)
				).detach().numpy()

		processed = y_hat[0, :-1]
		if use_softmax:
			processed = softmax(y_hat[0, :-1])

		thresholded = []
		predicted_values = []

		for k in Ks:
			threshold = sorted(processed, reverse=True)[k]
			array = processed.copy()
			array[array < threshold] = 0

			predicted_value = self.__get_prediction_value(array)
			predicted_values.append(predicted_value)

			thresholded.append(array)

		for k, arr, value in zip(Ks, thresholded, predicted_values):
			plt.figure()
			plt.title(f"{title}\nK: {k}, Predicted value: {value}\nCurrent Price: {sequence[-1]}")
			plt.plot(arr)
			plt.pause(0.01)

	def test_call_from_state(self):

		self.__call_from_state(
			self.tom_model,
			"/home/abrehamatlaw/Downloads/Compressed/results_10/graph_dumps/1736044405.169263",
			path=[0, 0],
			Ks=[429]
		)
		plt.show()

	def test_multiple(self):

		CONTAINER_PATH = "/home/abrehamatlaw/Downloads/Compressed/results_8/graph_dumps"

		START = 47
		BOUNDS = (START, START + 12)

		DUMP_PATHS = [
			os.path.join(CONTAINER_PATH, filename)
			for filename in sorted(os.listdir(CONTAINER_PATH))
		][BOUNDS[0]:BOUNDS[1]]

		for i, path in enumerate(DUMP_PATHS):
			print(f"Plotting: {path}, idx: {BOUNDS[0] + i}")
			self.__call_from_state(self.tom_model, path, title=f"State: {BOUNDS[0] + i}")

		plt.show()

	def __get_prediction_value(self, array: np.ndarray) -> float:
		return np.sum(self.bounds_centered * array[:-1] / np.sum(array[:-1]))

	def __continue_sequence(self, sequence: np.ndarray) -> np.ndarray:
		X = np.expand_dims(
			np.concatenate([
				sequence,
				np.zeros(Config.AGENT_MODEL_EXTRA_LEN)
			]),
			axis=0
		).astype(np.float32)

		with torch.no_grad():
			y_hat = self.tom_model(torch.from_numpy(X)).squeeze().detach().numpy()
			y_hat = softmax(y_hat[:-1])

		prediction = self.__get_prediction_value(y_hat)
		value = prediction * sequence[-1]
		return np.concatenate((sequence, np.array([value])), axis=0)[-len(sequence):]

	def __plot_from_sequence(self, sequence: np.ndarray):

		LENGTH = 3

		for i in range(LENGTH):
			sequence = self.__continue_sequence(sequence)

		plt.figure()
		plt.plot(sequence)
		plt.axvline(x=len(sequence)-LENGTH-1, color="red")

	def __get_state_sequence(self, path: str) -> np.ndarray:

		node, repo = stats.load_node_repo(
			path
		)
		state = repo.retrieve(node.id)
		return state.market_state.get_state_of("AUD", "USD")

	def test_plot_sequence_from_state(self):

		sequence = self.__get_state_sequence("/home/abrehamatlaw/Downloads/Compressed/results_10/graph_dumps/1736044405.169263")

		self.__plot_from_sequence(sequence)

		plt.show()

	def test_live_prediction(self):
		sequence = self.__get_state_sequence(
			"/home/abrehamatlaw/Downloads/Compressed/results_10/graph_dumps/1736044405.169263"
		)
		live_predictor = LivePredictor("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-6-tot.zip")

		live_predictor.start()
