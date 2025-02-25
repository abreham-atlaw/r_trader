import torch
import torch.nn as nn

from core.utils.research.model.model.ensemble.stacked import LinearMSM


class Linear3dMSM(LinearMSM):

	def collapse(self, x, y):
		if self.collapse_layer is None:
			self.collapse_layer = nn.Linear(x.shape[1], y.shape[1]*y.shape[2], bias=self.bias)
		out = self.collapse_layer(x)
		out = torch.reshape(out, y.shape)
		return out
