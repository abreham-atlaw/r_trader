import unittest

import numpy as np
import torch

from core import Config
from core.utils.research.losses import MSCECrossEntropyLoss



class MultiLossTest(unittest.TestCase):

	def test_functionality(self):
		classes = (np.arange(5) + 10).astype(np.float32)

		msce = MSCECrossEntropyLoss(classes, weights=[0.2, 0.8])

		y = torch.from_numpy(np.array([[0, 1, 0, 0, 0]]).astype(np.float32))
		predictions = torch.from_numpy(np.array([
			[[1 if i == j else 0 for j in range(len(classes))]]
			for i in range(len(classes))
		]).astype(np.float32))

		losses = [msce(y, predictions[i]) for i in range(len(classes))]
		self.assertIsNotNone(losses)
		# self.assertEqual(losses[1], 0)
		self.assertEqual(losses[0], losses[2])
		self.assertLess(losses[2], losses[3])
		self.assertLess(losses[3], losses[4])

		loss = msce(torch.from_numpy(np.repeat(y.numpy(), axis=0, repeats=len(classes))), torch.squeeze(predictions))
		self.assertEqual(loss, torch.mean(torch.Tensor(losses)))

	def test_percentage(self):
		classes = (np.arange(4) + 10).astype(np.float32)

		msce = MSCECrossEntropyLoss(classes)

		y = torch.from_numpy(np.array([[0, 1, 0, 0]]).astype(np.float32))
		predictions = torch.from_numpy(np.array([
			[[0.3, 0.5, 0.1, 0.1]],
			[[0.1, 0.8, 0.1, 0.0]],
			[[0.4, 0.2, 0.4, 0.0]],
			[[0.0, 1.0, 0.0, 0.0]]
		]).astype(np.float32))

		losses = [msce(y, predictions[i]) for i in range(len(classes))]
		self.assertIsNotNone(losses)
		self.assertLess(losses[1], losses[0])
		self.assertLess(losses[0], losses[2])
		self.assertLess(losses[3], losses[1])

