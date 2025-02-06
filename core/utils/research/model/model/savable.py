import importlib
import typing
from abc import ABC, abstractmethod

import torch
from torch import nn

from lib.utils.logger import Logger


class SpinozaModule(nn.Module, ABC):

	def __init__(self, *args, input_size=None, output_size=None, auto_build=True, **kwargs):
		super().__init__(*args, **kwargs)
		self.built = False

		if isinstance(input_size, int):
			input_size = (None, input_size)

		self.output_size = output_size
		self.input_size = input_size
		if input_size is not None and auto_build:
			self.init()
		self.__state_dict_params = None

	def init(self):
		init_data = torch.rand((1,) + self.input_size[1:])
		out = self(init_data)
		self.output_size = (None,) + out.size()[1:]

	def _build(self, input_size: torch.Size):
		Logger.info(f"[{self.__class__.__name__}] Building...")
		self.input_size = (None, ) + input_size[1:]
		self.build(input_size)
		self.built = True
		if self.__state_dict_params is not None:
			Logger.info(f"[{self.__class__.__name__}] Loading state dict...")
			self.load_state_dict(*self.__state_dict_params[0], ** self.__state_dict_params[1])
			self.__state_dict_params = None

	@staticmethod
	def _get_input_size(*args, **kwargs) -> torch.Size:
		return args[0].size()

	def build(self, input_size: torch.Size):
		pass

	@abstractmethod
	def call(self, *args, **kwargs) -> torch.Tensor:
		pass

	def forward(self, *args, **kwargs) -> torch.Tensor:
		if not self.built:
			self._build(self._get_input_size(*args, **kwargs))
		return self.call(*args, **kwargs)

	def load_state_dict_lazy(self, *args, **kwargs):
		self.__state_dict_params = args, kwargs
		self.built = False

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


