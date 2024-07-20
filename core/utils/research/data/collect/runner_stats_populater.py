import os.path
import typing
import uuid
from datetime import datetime

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.Config import MODEL_SAVE_EXTENSION
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats
from core.utils.research.model.model.wrapped import WrappedModel
from core.utils.research.training.trainer import Trainer
from lib.utils.file_storage import FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class RunnerStatsPopulater:

	def __init__(
			self,
			repository: RunnerStatsRepository,
			in_filestorage: FileStorage,
			out_filestorage: FileStorage,
			dataloader: DataLoader,
			in_path: str,
			tmp_path: str = "./",
			ma_window: int = 10,
			device: typing.Optional[str] = None
	):
		self.__in_filestorage = in_filestorage
		self.__out_filestorage = out_filestorage
		self.__in_path = in_path
		self.__repository = repository
		self.__tmp_path = tmp_path
		self.__dataloader = dataloader
		self.__ma_window = ma_window
		self.__device = device

	def __generate_tmp_path(self, ex=MODEL_SAVE_EXTENSION):
		return os.path.join(self.__tmp_path, f"{datetime.now().timestamp()}.{ex}")

	def __evaluate_model(self, model: nn.Module) ->float:
		print("[+]Evaluating Model")
		trainer = Trainer(model, cls_loss_function=nn.CrossEntropyLoss(), optimizer=Adam(model.parameters()))
		return trainer.validate(self.__dataloader)

	def __prepare_model(self, model: nn.Module) -> nn.Module:
		print("[+]Preparing Model")
		return WrappedModel(model, window_size=self.__ma_window)

	def _upload_model(self, model: nn.Module) -> str:
		print("[+]Exporting Model")
		path = self.__generate_tmp_path()
		ModelHandler.save(model, path)
		self.__out_filestorage.upload_file(path)
		# return self.__out_filestorage.get_url(os.path.basename(path))
		return os.path.basename(path)

	def __generate_id(self) -> str:
		return uuid.uuid4().hex

	def _process_model(self, path: str):
		local_path = self.__generate_tmp_path()
		self.__in_filestorage.download(path, local_path)
		model = ModelHandler.load(local_path, device=self.__device)

		loss = self.__evaluate_model(model)
		model = self.__prepare_model(model)
		upload_path = self._upload_model(model)
		id = self.__generate_id()

		stats = RunnerStats(
			id=id,
			model_name=upload_path,
			model_loss=loss
		)
		self.__repository.store(stats)

	def start(self):
		files = self.__in_filestorage.listdir(self.__in_path)
		for i, file in enumerate(files):
			try:
				self._process_model(file)
			except Exception as ex:
				print(f"[-]Error Occurred processing {file}\n{ex}")
			print(f"{(i+1)*100/len(files) :.2f}", end="\r")
