import typing

import torch


class TensorCollection:

	def __init__(self, tensors: typing.List[torch.Tensor]):
		self.tensors = tensors

	def __getitem__(self, idx: int) -> torch.Tensor:
		return self.tensors[idx]

	def __len__(self) -> int:
		return len(self.tensors)

	def to(self, device: torch.device) -> 'TensorCollection':
		return TensorCollection([t.to(device) for t in self.tensors])

	def type(self, dtype: torch.dtype) -> 'TensorCollection':
		return TensorCollection([t.type(dtype) for t in self.tensors])
