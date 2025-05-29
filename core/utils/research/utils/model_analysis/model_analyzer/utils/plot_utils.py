import math
import typing

import matplotlib.pyplot as plt
import torch
import numpy as np


class PlotUtils:

	@staticmethod
	def __to_numpy(X: typing.Union[torch.Tensor, np.ndarray]) -> np.ndarray:
		if isinstance(X, torch.Tensor):
			return X.detach().cpu().numpy()
		return X

	@staticmethod
	def __generate_X(y: np.ndarray) -> np.ndarray:
		X = np.arange(y.shape[-1])
		for i in range(len(y.shape) - 1):
			X = np.repeat(np.expand_dims(X, axis=0), repeats=y.shape[-(i+2)], axis=0)
		return X

	@staticmethod
	def __plot_subplot(
			X: np.ndarray,
			y: np.ndarray,
			title: str,
			start_idx: int
	):
		plt.grid(True)
		for i in range(y.shape[0]):
			plt.plot(X[i], y[i], label=f"{start_idx + i}")
		plt.legend()
		plt.title(title)

	@staticmethod
	def __plot_figure(
			X: np.ndarray,
			y: np.ndarray,
			fig_size: typing.Tuple[int, int],
			title: str,
			max_plots: int,
			cols: int,
	):
		subplots = math.ceil(y.shape[0] / max_plots)
		cols = min(cols, subplots)
		rows = math.ceil(subplots / cols)

		plt.figure(figsize=(fig_size[0]*cols, fig_size[1]*rows))
		plt.title(title)

		for i in range(subplots):
			plt.subplot(rows, cols, i + 1)
			subplot_range = i*max_plots, min(y.shape[0], (i+1)*max_plots)
			PlotUtils.__plot_subplot(
				X[subplot_range[0]:subplot_range[1]],
				y[subplot_range[0]:subplot_range[1]],
				f"{title} ({subplot_range[0]}-{subplot_range[1]}) ",
				start_idx=subplot_range[0]
			)

	@staticmethod
	def plot(
			y: typing.Union[torch.Tensor, np.ndarray],
			X: typing.Union[np.ndarray, torch.Tensor] = None,
			title: str = "",
			fig_size: typing.Tuple[int, int] = None,
			max_plots: int = 5
	):
		y = PlotUtils.__to_numpy(y)

		while len(y.shape) < 3:
			y = np.expand_dims(y, axis=0)

		if X is None:
			X = PlotUtils.__generate_X(y)

		if fig_size is None:
			fig_size = 20, 10

		X = PlotUtils.__to_numpy(X)

		for i in range(y.shape[0]):
			PlotUtils.__plot_figure(
				X[i],
				y[i],
				fig_size=fig_size,
				title=f"{title} ({i}) ",
				max_plots=max_plots,
				cols=2
			)

		plt.pause(0.1)
