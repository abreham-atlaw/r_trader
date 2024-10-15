import unittest

import numpy as np
import torch

from core import Config
from core.utils.research.losses import MeanSquaredClassError
from core.utils.research.losses.rwl import ReverseMAWeightLoss


class ReverseWeightLossTest(unittest.TestCase):

	def test_functionality(self):

		classes = (np.arange(5) + 10).astype(np.float32)

		loss_fn = ReverseMAWeightLoss(window_size=2, softmax=False)

		y = torch.from_numpy(np.array([[0, 1, 0, 0, 0]]).astype(np.float32))
		predictions = torch.from_numpy(np.array([
			[[1 if i == j else 0 for j in range(len(classes))]]
			for i in range(len(classes))
		]).astype(np.float32))

		losses = [loss_fn(y, predictions[i]) for i in range(len(classes))]
		self.assertIsNotNone(losses)
		self.assertEqual(losses[1], 0)
		self.assertEqual(losses[0], losses[2])
		self.assertLess(losses[2], losses[3])

		loss = loss_fn(torch.from_numpy(np.repeat(y.numpy(), axis=0, repeats=len(classes))), torch.squeeze(predictions))
		self.assertEqual(loss, torch.mean(torch.Tensor(losses)))

	def test_batch(self):

		classes = (np.arange(5) + 10).astype(np.float32)

		y = torch.from_numpy(np.array([
			[1 if i == j else 0 for j in range(len(classes))]
			for i in range(len(classes))
		]).astype(np.float32))

		predictions = torch.rand((len(classes), len(classes)))

		loss_fn = ReverseMAWeightLoss(window_size=2)
		batched_loss = loss_fn(y, predictions)

		unbatched_loss = torch.mean(torch.Tensor([loss_fn(y[i:i+1], predictions[i:i+1]) for i in range(len(classes))]))

		self.assertTrue(torch.all(batched_loss == unbatched_loss))