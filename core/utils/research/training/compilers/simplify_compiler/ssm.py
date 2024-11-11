from torch import nn

from core.utils.research.model.model.savable import SpinozaModule


class SimplifiedSpinozaModule(nn.Module):

	def __init__(self):
		super().__init__()
