import re
import typing
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from lib.utils.logger import Logger


class ModelMigrator(ABC):

	def __init__(self):
		self.__pattern_mapping = self._get_pattern_generator_mapping()

	@abstractmethod
	def _create_model(self, og_model: nn.Module) -> nn.Module:
		pass

	@abstractmethod
	def _get_pattern_generator_mapping(self) -> typing.Dict[str, typing.Callable]:
		pass

	def __generate_key(self, key: str) -> str:
		for pattern, generator in self.__pattern_mapping.items():
			if re.match(pattern, key):
				return generator(key)
		raise KeyMatchNotFoundException(key)

	def _get_state_dict_mapping(self, og_state_dict_keys: typing.List[str]) -> typing.Dict[str, str]:
		return {
			og_key: self.__generate_key(og_key)
			for og_key in og_state_dict_keys
		}

	def _migrate_state_dict(
			self,
			og_state_dict: typing.Dict[str, torch.Tensor],
			state_dict: typing.Dict[str, torch.Tensor]
	):
		mapping = self._get_state_dict_mapping(list(og_state_dict.keys()))

		for old_key, new_key in mapping.items():
			if new_key not in state_dict:
				Logger.warning(f"{new_key} not in state dict; Skipping. Present Keys are: {state_dict.keys()}")
				continue
			state_dict[new_key] = og_state_dict[old_key]

	def migrate(self, model: nn.Module) -> nn.Module:
		new_model = self._create_model(model)

		og_state_dict = model.state_dict()
		new_state_dict = new_model.state_dict().copy()

		self._migrate_state_dict(og_state_dict, new_state_dict)

		new_model.load_state_dict(new_state_dict)

		return new_model


class KeyMatchNotFoundException(Exception):

	def __init__(self, key: str):
		super().__init__(f"Key {key} not found in state dict")
