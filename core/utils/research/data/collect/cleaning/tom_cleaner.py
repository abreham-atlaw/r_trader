import random
import typing
from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn

from core import Config
from core.di import ResearchProvider, ServiceProvider
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from core.utils.research.model.model.utils import TransitionOnlyModel
from lib.utils.file_storage import FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class TomCleaner:

	def __init__(
			self,
			X: typing.Union[torch.Tensor, np.ndarray],
			threshold: float,
			date_threshold: datetime = None,
			repository: RunnerStatsRepository = None,
			fs: FileStorage = None,
			in_path="/",
			tmp_path: str = "/tmp/",

	):

		if isinstance(X, np.ndarray):
			X = torch.from_numpy(X)
		self.__X = X
		self.__threshold = threshold
		self.__tmp_path = tmp_path

		if date_threshold is None:
			date_threshold = datetime.now()
		self.__date_threshold = date_threshold

		if repository is None:
			repository = ResearchProvider.provide_runner_stats_repository()
		self.__repository = repository

		if fs is None:
			fs = ServiceProvider.provide_file_storage()
		self.__fs = fs

		self.__tmp_path = tmp_path
		self.__in_path = in_path

	def __generate_download_path(self, stat: RunnerStats) -> str:
		return os.path.join(self.__tmp_path, stat.model_name)

	def __call_model(self, model: nn.Module) -> torch.Tensor:
		return model(self.__X)

	def __compare_models(
			self,
			a: nn.Module,
			b: nn.Module
	) -> float:
		return (self.__call_model(a) - self.__call_model(b)).abs().mean().item()

	def evaluate_model(self, model) -> float:
		return self.__compare_models(
			model,
			TransitionOnlyModel(model, extra_len=Config.AGENT_MODEL_EXTRA_LEN)
		)

	def __is_valid_model(self, model: nn.Module) -> bool:
		print(f"[+] Validating {model.__class__.__name__}...")
		difference = self.evaluate_model(model)
		print(f"[+] Difference: {difference}")
		return difference <= self.__threshold

	def __load_model(self, stat: RunnerStats) -> nn.Module:
		print(f"[+] Loading {stat.model_name} ...")
		path = self.__generate_download_path(stat)
		self.__fs.download(
			os.path.join(self.__in_path, stat.model_name),
			download_path=path
		)
		return ModelHandler.load(path).to('cpu')

	def __clear_pl(self, stat: RunnerStats):
		print(f"[+] Cleaning...")

		invalid_idxs = [
			i
			for i, timestamp in enumerate(stat.session_timestamps)
			if timestamp < self.__date_threshold
		]

		stat.profits = [
			stat.profits[i]
			for i in range(len(stat.profits))
			if i not in invalid_idxs
		]
		stat.session_timestamps = [
			stat.session_timestamps[i]
			for i in range(len(stat.session_timestamps))
			if i not in invalid_idxs
		]

		self.__repository.store(stat)

	def __process_stat(self, stat: RunnerStats):
		print(f"[+] Processing {stat.id} ...")
		model = self.__load_model(stat)

		if self.__is_valid_model(model):
			print(f"[+] {stat.id} is valid")
			return

		self.__clear_pl(stat)

	def start(self, stats: typing.List[RunnerStats] = None):
		if stats is None:
			stats = self.__repository.retrieve_all()

		random.shuffle(stats)

		print(f"[+] Processing {len(stats)} stats")

		for i, stat in enumerate(stats):
			try:
				self.__process_stat(stat)
			except Exception as e:
				print(f"[-] Failed to process {stat.id}: {e}")
			print(f"[+] \nProgress: {(i + 1) * 100 / len(stats):.2f}% ...")

		print(f"[+] Done!")