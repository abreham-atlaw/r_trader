import typing

from torch import nn as nn

from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.cnn.residual_block import ResidualBlock


class ResNet(CNN):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _build_conv_layers(
			self,
			channels: typing.List[int],
			kernel_sizes: typing.List[int],
			stride: typing.List[int],
			padding: int
	) -> typing.List[nn.Module]:
		return [
			ResidualBlock(
				in_channels=channels[i],
				out_channels=channels[i + 1],
				kernel_size=kernel_sizes[i],
				padding=padding,
				init_fn=None,
				norm=False
			)
			for i in range(len(channels) - 1)
		]
