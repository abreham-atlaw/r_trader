from typing import *

from datetime import datetime

from .data.models import Config, Progress
from lib.utils.logger import Logger


class OptimizationCacher:

	def __init__(self):
		pass

	def __construct_config(self, config_dict: Dict, value=None) -> Config:
		config = Config(
			id=None,
			seq_len=config_dict.get("seq_len"),
			loss=config_dict.get("loss"),
			optimizer=config_dict.get("optimizer"),
			hidden_activation=config_dict.get("hidden_activation"),
			delta=config_dict.get("delta"),
			percentage=config_dict.get("percentage"),
			average_window=config_dict.get("average_window"),
			value=value
		)
		config.set_hidden_layers(config_dict.get("hidden_layers"))
		return config

	@Logger.logged_method
	def get_value(self, config_dict: Dict) -> Union[float, None]:
		config = self.__construct_config(config_dict)
		return config.get_value()

	@Logger.logged_method
	def cache(self, config_dict: Dict, value: float):
		config = self.__construct_config(config_dict, value)
		config.save(commit=True)
		progress: Progress = Progress.get_by_config_id(config.id)
		if progress is not None:
			progress.set_done(commit=True)

	@Logger.logged_method
	def is_locked(self, config_dict: Dict) -> bool:
		config = self.__construct_config(config_dict)
		config_id = config.get_id()
		if config_id is None:
			return False

		progress: Progress = Progress.get_by_config_id(config_id)
		if progress is None:
			return False
		return not progress.done

	@Logger.logged_method
	def lock(self, config_dict: Dict, user: str):
		config = self.__construct_config(config_dict)
		config.save()
		progress = Progress(
			id=None,
			config_id=config.id,
			user=user,
			start_datetime=datetime.now()
		)
		progress.save(commit=True)
