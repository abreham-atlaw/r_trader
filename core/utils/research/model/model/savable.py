import importlib
import typing
from abc import ABC, abstractmethod

from torch import nn


class SavableModule(nn.Module, ABC):

	@abstractmethod
	def export_config(self) -> typing.Dict[str, typing.Any]:
		pass

	@classmethod
	def import_config(cls, config) -> typing.Dict[str, typing.Any]:
		if config.get('hidden_activation'):
			hidden_activation_module = importlib.import_module('torch.nn')  # replace with the actual module
			config['hidden_activation'] = getattr(hidden_activation_module, config['hidden_activation'])()
		if config.get('init_fn'):
			init_fn_module = importlib.import_module('torch.nn.init')  # replace with the actual module
			config['init_fn'] = getattr(init_fn_module, config['init_fn'])
		return config


