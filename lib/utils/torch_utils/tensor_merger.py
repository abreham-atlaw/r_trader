import typing

import torch

import pickle

from lib.utils.logger import Logger
from .slice_serializer import SliceSerializer


class TensorMerger:

	def __init__(self, use_shared_dims: bool = False):
		self.slices = None
		self.target_shape = None
		self.use_shared_dims = use_shared_dims

	def __shared_dims_init(self, tensors: typing.List[torch.Tensor]):

		slices = [[] for _ in range(len(tensors))]
		target_shape = []

		concat_axis = None

		for i in range(tensors[0].dim()):

			is_common_dim = concat_axis is None and i < tensors[0].dim() - 1 and False not in [
				tensors[0].shape[i] == t.shape[i]
				for t in tensors
			]

			is_concat_axis = concat_axis is None and not is_common_dim
			if is_concat_axis:
				concat_axis = i

			axis_sum = 0

			for j in range(len(tensors)):

				if is_common_dim:
					tensor_slice = slice(None)
				elif is_concat_axis:
					tensor_slice = slice(axis_sum, axis_sum + tensors[j].shape[i])
					axis_sum += tensors[j].shape[i]
				else:
					tensor_slice = slice(0, tensors[j].shape[i])

				slices[j].append(tensor_slice)

			if is_common_dim:
				target_shape.append(None)
			elif is_concat_axis:
				target_shape.append(axis_sum)
			else:
				target_shape.append(max([t.shape[i] for t in tensors]))

		slices = [tuple(s) for s in slices]
		target_shape = tuple(target_shape)

		return slices, target_shape

	def __legacy_init(self, tensors: typing.List[torch.Tensor]):
		slices = [
			(i, *tuple(slice(0, size) for size in t.shape))
			for i, t in enumerate(tensors)
		]
		target_shape = (len(tensors),) + tuple([
			max([
				t.shape[i] for t in tensors
			])
			for i in range(tensors[0].dim())
		])
		return slices, target_shape

	def __init_slices(self, tensors: typing.List[torch.Tensor]):
		if self.slices is not None:
			return

		init_fn: typing.Callable = self.__shared_dims_init if self.use_shared_dims else self.__legacy_init
		self.slices, self.target_shape = init_fn(tensors)

		Logger.info(f"Initialized TensorMerger with target shape: {self.target_shape} and slices: {self.slices}")

	def __get_target_shape(self, tensors: typing.List[torch.Tensor]):
		return [
			self.target_shape[i] if self.target_shape[i] is not None else tensors[0].shape[i]
			for i in range(tensors[0].dim())
		]

	def merge(self, tensors: typing.List[torch.Tensor]) -> torch.Tensor:

		if False in [tensors[0].dim() == t.dim() for t in tensors[1:]]:
			raise ValueError("Cannot merge tensors with different number of dimensions")

		self.__init_slices(tensors)

		out = torch.zeros(self.__get_target_shape(tensors), dtype=tensors[0].dtype, device=tensors[0].device)

		for slices, t in zip(self.slices, tensors):
			out[slices] = t

		return out

	def split(self, tensor: torch.Tensor) -> typing.List[torch.Tensor]:

		if self.slices is None:
			raise ValueError("Slices not initialized. Call merge before split")

		target_slices = self.slices
		if tensor.dim() == len(self.target_shape) + 1:
			target_slices = [
				(slice(0, tensor.shape[0]),) + slices
				for slices in target_slices
			]

		return [
			tensor[slices]
			for slices in target_slices
		]

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return {
			"slices": self.slices,
			"target_shape": self.target_shape
		}

	def save_config(self, path: str):
		with open(path, "wb") as f:
			pickle.dump(self.export_config(), f)

	def import_config(self, config: typing.Dict[str, typing.Any]):
		self.slices = config["slices"]
		self.target_shape = config["target_shape"]

	def load_config(self, path: str):
		with open(path, "rb") as f:
			return self.import_config(pickle.load(f))

	@staticmethod
	def load_from_config(path: str) -> 'TensorMerger':
		merger = TensorMerger()
		merger.load_config(path)
		return merger
