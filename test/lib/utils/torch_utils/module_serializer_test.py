import unittest

import torch
from torch import nn

from lib.utils.torch_utils.module_serializer import TorchModuleSerializer


class ModuleSerializerTest(unittest.TestCase):

	def setUp(self):
		self.serializer = TorchModuleSerializer()
		self.module = nn.LeakyReLU(negative_slope=0.1)

	def test_functionality(self):
		X = torch.rand((2, 10)) - 0.5

		serialized = self.serializer.serialize(self.module)
		deserialized: nn.Module = self.serializer.deserialize(serialized)

		y = self.module(X)
		y_hat = deserialized(X)

		self.assertTrue(torch.allclose(y, y_hat))
