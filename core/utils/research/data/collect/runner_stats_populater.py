import os.path
import typing
import uuid
from datetime import datetime
import random

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from core import Config
from core.Config import MODEL_SAVE_EXTENSION
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats
from core.utils.research.losses import WeightedMSELoss, MSCECrossEntropyLoss, ReverseMAWeightLoss, \
	MeanSquaredClassError, PredictionConfidenceScore, OutputClassesVariance, OutputBatchVariance
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

	def __generate_tmp_path(self, ex=MODEL_SAVE_EXTENSION):
		return os.path.join(self.__tmp_path, f"{datetime.now().timestamp()}.{ex}")

	def __evaluate_model_loss(self, model: nn.Module, cls_loss_fn: typing.Optional[nn.Module]) -> float:
		print("[+]Evaluating Model")
		trainer = Trainer(model, cls_loss_function=cls_loss_fn, reg_loss_function=nn.MSELoss(), optimizer=Adam(model.parameters()))
		loss = trainer.validate(self.__dataloader)
		if isinstance(loss, list):
			return loss[-1]
		return loss

	def __evaluate_model(self, model: nn.Module) -> typing.Tuple[float, ...]:
		return tuple([
			self.__evaluate_model_loss(
				model,
				loss
			)
			for loss in [
				nn.CrossEntropyLoss(),
				WeightedMSELoss(len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1),
				MeanSquaredClassError(
					Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
					Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON
				),
				ReverseMAWeightLoss(window_size=5, softmax=True),
				ReverseMAWeightLoss(window_size=10, softmax=True),
				ReverseMAWeightLoss(window_size=20, softmax=True),
				ReverseMAWeightLoss(window_size=40, softmax=True),
				PredictionConfidenceScore(softmax=True),
				OutputClassesVariance(softmax=True),
				OutputBatchVariance(softmax=True),
			]
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
		local_path = self.__download_model(path)
		model = TemperatureScalingModel(
			ModelHandler.load(local_path),
			temperature=temperature
		)
		print(f"[+]Evaluating...")
		losses = self.__evaluate_model(model)
		id = self.__generate_id(path, temperature)

		stats = self.__repository.retrieve(id)
		if stats is None:
			print("[+]Creating...")
			stats = RunnerStats(
				id=id,
				model_name=os.path.basename(path),
				model_losses=losses,
				session_timestamps=[],
				temperature=temperature
			)
		else:
			print("[+]Updating...")
			stats.model_losses = losses
		self.__repository.store(stats)
		self.__junk.add(local_path)

	def __is_processed(self, file_path: str, temperature: float) -> bool:
		stat = self.__repository.retrieve(
			self.__generate_id(file_path, temperature=temperature)
		)
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
						print(f"[+]Skipping {file}. Already Processed")
						continue
					self._process_model(file, temperature)
				except Exception as ex:
					print(f"[-]Error Occurred processing {file}\n{ex}")
					if self.__raise_exception:
						raise ex

			print(f"{(i+1)*100/len(files) :.2f}", end="\r")

		self.__clean_junk()
