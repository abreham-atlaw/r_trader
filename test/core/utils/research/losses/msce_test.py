import unittest

import numpy as np
import torch

from core import Config
from core.utils.research.losses import MeanSquaredClassError


class MSCETest(unittest.TestCase):

	def test_functionality(self):

		classes = (np.arange(5) + 10).astype(np.float32)

		msce = MeanSquaredClassError(classes)

		y = torch.from_numpy(np.array([[0, 1, 0, 0, 0]]).astype(np.float32))
		predictions = torch.from_numpy(np.array([
			[[1 if i == j else 0 for j in range(len(classes))]]
			for i in range(len(classes))
		]).astype(np.float32))

		losses = [msce(y, predictions[i]) for i in range(len(classes))]
		self.assertIsNotNone(losses)
		self.assertEqual(losses[1], 0)
		self.assertEqual(losses[0], losses[2])
		self.assertLess(losses[2], losses[3])
		self.assertLess(losses[3], losses[4])

		loss = msce(y, torch.squeeze(predictions))
		self.assertEqual(loss, torch.mean(torch.Tensor(losses)))

	def test_actual(self):

		classes = np.array(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND)

		msce = MeanSquaredClassError(classes, epsilon=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON)

		y = torch.zeros((1, len(classes)+1), dtype=torch.float32)
		y[0, 3] = 1
		predictions = torch.from_numpy(np.array([
			[[1 if i == j else 0 for j in range(len(classes)+1)]]
			for i in range(len(classes)+1)
		]).astype(np.float32))

		losses = [msce(y, predictions[i]) for i in range(len(classes))]
		self.assertNotEqual(losses, 0)
