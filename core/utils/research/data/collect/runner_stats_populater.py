import os.path
import typing
import uuid
from datetime import datetime
import random

import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from core import Config
from core.Config import MODEL_SAVE_EXTENSION
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats
from core.utils.research.losses import WeightedMSELoss, MSCECrossEntropyLoss, ReverseMAWeightLoss, \
	MeanSquaredClassError, PredictionConfidenceScore, OutputClassesVariance, OutputBatchVariance, ProximalMaskedLoss, \
	OutputBatchClassVariance
from core.utils.research.model.model.utils import TemperatureScalingModel
from core.utils.research.training.trainer import Trainer
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.file_storage import FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class RunnerStatsPopulater:

	def __init__(
			self,
			repository: RunnerStatsRepository,
			in_filestorage: FileStorage,
			dataloader: DataLoader,
			in_path: str,
			tmp_path: str = "./",
			ma_window: int = 10,
			shuffle_order: bool = True,
			raise_exception: bool = False,
			exception_exceptions: typing.List[typing.Type] = None,
			temperatures: typing.Tuple[float,...] = (1.0,)
	):
		self.__in_filestorage = in_filestorage
		self.__in_path = in_path
		self.__repository = repository
		self.__tmp_path = tmp_path
		self.__dataloader = dataloader
		self.__ma_window = ma_window
		self.__shuffle_order = shuffle_order
		self.__raise_exception = raise_exception
		self.__temperatures = temperatures
		self.__junk = set([])

		if exception_exceptions is None:
			exception_exceptions = []
		self.__exception_exceptions = exception_exceptions
		self.__loss_functions = self.__get_evaluation_loss_functions()

	def __generate_tmp_path(self, ex=MODEL_SAVE_EXTENSION):
		return os.path.join(self.__tmp_path, f"{datetime.now().timestamp()}.{ex}")

	def __evaluate_model_loss(self, model: nn.Module, cls_loss_fn: typing.Optional[nn.Module]) -> float:
		print(f"[+]Evaluating Model with {cls_loss_fn.__class__.__name__}")
		trainer = Trainer(model, cls_loss_function=cls_loss_fn, reg_loss_function=nn.MSELoss(), optimizer=Adam(model.parameters()))
		loss = trainer.validate(self.__dataloader)
		if isinstance(loss, list):
			return loss[-1]
		return loss

	def __get_evaluation_loss_functions(self):
		return [
				nn.CrossEntropyLoss(),
				ProximalMaskedLoss(
					n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
					softmax=True,
				),
				MeanSquaredClassError(
					Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
					Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON
				),
				ReverseMAWeightLoss(window_size=10, softmax=True),
				PredictionConfidenceScore(softmax=True),
				OutputClassesVariance(softmax=True),
				OutputBatchVariance(softmax=True),
				OutputBatchClassVariance(
					np.array(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND),
					epsilon=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON,
				),
				ProximalMaskedLoss(
					n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
					softmax=True,
					p=0.5
				),
				ProximalMaskedLoss(
					n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
					softmax=True,
					p=0.25
				),
				ProximalMaskedLoss(
					n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
					softmax=True,
					p=0.1
				),
				ProximalMaskedLoss(
					n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
					softmax=True,
					weights=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_WEIGHTS
				)
			]

	def __evaluate_model(self, model: nn.Module, current_losses) -> typing.Tuple[float, ...]:
		return tuple([
			self.__evaluate_model_loss(
				model,
				loss
			) if current_losses is None or current_losses[i] == 0.0 else current_losses[i]
			for i, loss in enumerate(self.__loss_functions)
		])

	def __prepare_model(self, model: nn.Module) -> nn.Module:
		return model

	def __clean_junk(self):
		print(f"[+]Cleaning Junk...")
		for path in self.__junk:
			os.system(f"rm {os.path.abspath(path)}")

	def __generate_id(self, file_path: str, temperature: float) -> str:
		id = os.path.basename(file_path).replace(MODEL_SAVE_EXTENSION, "")
		if temperature != 1.0:
			id = f"{id}-(T={temperature})"
		return id

	@CacheDecorators.cached_method()
	def __download_model(self, path: str):
		print(f"[+]Downloading...")
		local_path = self.__generate_tmp_path()
		self.__in_filestorage.download(path, local_path)
		return local_path

	def _process_model(self, path: str, temperature: float):
		print(f"[+]Processing {path}(T={temperature})...")

		stat = self.__repository.retrieve(self.__generate_id(path, temperature))
		current_losses = stat.model_losses if stat is not None else None

		local_path = self.__download_model(path)
		model = TemperatureScalingModel(
			ModelHandler.load(local_path),
			temperature=temperature
		)

		if current_losses is not None and False not in [l == 0 for l in current_losses]:
			current_losses = None

		print(f"[+]Evaluating...")
		losses = self.__evaluate_model(model, current_losses)
		id = self.__generate_id(path, temperature)

		stats = self.__repository.retrieve(id)
		if stats is None:
			print("[+]Creating...")
			stats = RunnerStats(
				id=id,
				model_name=os.path.basename(path),
				session_timestamps=[],
				temperature=temperature
			)
			stats.model_losses = losses
		else:
			print("[+]Updating...")
			stats.model_losses = losses
		self.__repository.store(stats)
		self.__junk.add(local_path)

	def __is_processed(self, file_path: str, temperature: float) -> bool:
		stat = self.__repository.retrieve(
			self.__generate_id(file_path, temperature=temperature)
		)

		if stat is not None and len(stat.model_losses) < len(self.__loss_functions):
			stat.model_losses += tuple([0.0] * (len(self.__loss_functions) - len(stat.model_losses)))

		return stat is not None and 0.0 not in stat.model_losses

	def start(self, replace_existing: bool = False):
		files = self.__in_filestorage.listdir(self.__in_path)
		if self.__shuffle_order:
			print("[+]Shuffling Files")
			random.shuffle(files)
		for i, file in enumerate(files):
			for temperature in self.__temperatures:
				try:
					if self.__is_processed(file, temperature) and not replace_existing:
						print(f"[+]Skipping {file}(T={temperature}). Already Processed")
						continue
					self._process_model(file, temperature)
				except Exception as ex:
					print(f"[-]Error Occurred processing {file}\n{ex}")
					if self.__raise_exception or True in [isinstance(ex, exception_class) for exception_class in self.__exception_exceptions]:
						raise ex

			print(f"{(i+1)*100/len(files) :.2f}", end="\r")

		self.__clean_junk()
