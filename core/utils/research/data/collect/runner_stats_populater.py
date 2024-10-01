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
from core.utils.research.training.trainer import Trainer
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
			raise_exception: bool = False
	):
		self.__in_filestorage = in_filestorage
		self.__in_path = in_path
		self.__repository = repository
		self.__tmp_path = tmp_path
		self.__dataloader = dataloader
		self.__ma_window = ma_window
		self.__shuffle_order = shuffle_order
		self.__raise_exception = raise_exception

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

	def __clean(self, local_path: str):
		os.system(f"rm {os.path.abspath(local_path)}")

	def __generate_id(self, file_path: str) -> str:
		return os.path.basename(file_path).replace(MODEL_SAVE_EXTENSION, "")

	def _process_model(self, path: str):
		local_path = self.__generate_tmp_path()
		self.__in_filestorage.download(path, local_path)
		model = ModelHandler.load(local_path)

		losses = self.__evaluate_model(model)
		id = self.__generate_id(path)

		stats = self.__repository.retrieve(id)
		if stats is None:
			print("[+]Creating...")
			stats = RunnerStats(
				id=id,
				model_name=os.path.basename(path),
				model_losses=losses,
				session_timestamps=[]
			)
		else:
			print("[+]Updating...")
			stats.model_losses = losses
		self.__repository.store(stats)
		self.__clean(local_path)

	def __is_processed(self, file_path: str) -> bool:
		stat = self.__repository.retrieve(self.__generate_id(file_path))
		return stat is not None and 0.0 not in stat.model_losses

	def start(self, replace_existing: bool = False):
		files = self.__in_filestorage.listdir(self.__in_path)
		if self.__shuffle_order:
			print("[+]Shuffling Files")
			random.shuffle(files)
		for i, file in enumerate(files):
			try:
				if self.__is_processed(file) and not replace_existing:
					print(f"[+]Skipping {file}. Already Processed")
					continue
				self._process_model(file)
			except Exception as ex:
				print(f"[-]Error Occurred processing {file}\n{ex}")
				if self.__raise_exception:
					raise ex
			print(f"{(i+1)*100/len(files) :.2f}", end="\r")
