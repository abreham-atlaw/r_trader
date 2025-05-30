import math
import typing

import matplotlib.pyplot as plt
import torch
import numpy as np


class PlotUtils:

	class Mode:
		PLOT = 0
		IMAGE = 1

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
			start_idx: int,
			mode: int
	):
		plt.grid(True)

		if mode == PlotUtils.Mode.PLOT:
			for i in range(y.shape[0]):
				plt.plot(X[i], y[i], label=f"{start_idx + i}")
			plt.legend()

		elif mode == PlotUtils.Mode.IMAGE:
			y = y[0]
			plt.imshow(y, aspect="auto", cmap="seismic", extent=[0, y.shape[1], y.shape[0], 0])
			plt.colorbar(label='Value')

		plt.title(title)

	@staticmethod
	def __plot_figure(
			X: np.ndarray,
			y: np.ndarray,
			fig_size: typing.Tuple[int, int],
			title: str,
			max_plots: int,
			cols: int,
			mode: int
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
				start_idx=subplot_range[0],
				mode=mode
			)

	@staticmethod
	def plot(
			y: typing.Union[torch.Tensor, np.ndarray],
			X: typing.Union[np.ndarray, torch.Tensor] = None,
			title: str = "",
			fig_size: typing.Tuple[int, int] = None,
			max_plots: int = 5,
			mode: int = Mode.PLOT
	):
		y = PlotUtils.__to_numpy(y)

		min_dims = 3

		if mode == PlotUtils.Mode.IMAGE:
			min_dims += 1
			max_plots = 1

		while len(y.shape) < min_dims:
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
				cols=2,
				mode=mode
			)

		plt.pause(0.1)
