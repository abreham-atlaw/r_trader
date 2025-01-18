import typing

import torch

from lib.utils.logger import Logger


class TensorMerger:

	def __init__(self):
		self.slices = None
		self.target_shape = None

	def __init_slices(self, tensors: typing.List[torch.Tensor]) -> typing.List[typing.Tuple[int, ...]]:
		if self.slices is not None:
			return
		self.original_shapes = [t.shape for t in tensors]
		self.slices = [
			(i, *tuple(slice(0, size) for size in t.shape))
			for i, t in enumerate(tensors)
		]
		self.target_shape = (len(tensors),) +tuple([
			max([
				t.shape[i] for t in tensors
			])
			for i in range(tensors[0].dim())
		])
		Logger.info(f"Initialized TensorMerger with target shape: {self.target_shape} and slices: {self.slices}")

	def merge(self, tensors: typing.List[torch.Tensor]) -> torch.Tensor:

		if False in [tensors[0].dim() == t.dim() for t in tensors[1:]]:
			raise ValueError("Cannot merge tensors with different number of dimensions")

		self.__init_slices(tensors)

		out = torch.zeros(self.target_shape, dtype=tensors[0].dtype, device=tensors[0].device)

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
