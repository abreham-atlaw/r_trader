from typing import *

from .data.models import Config
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
