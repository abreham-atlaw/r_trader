import unittest

import numpy as np
import torch

from core.utils.research.losses import ProximalMaskedLoss


class ProximalMaskedLossTest(unittest.TestCase):

	def test_functionality(self):
		classes = (np.arange(5) + 10).astype(np.float32)

		loss_fn = ProximalMaskedLoss(
			n=len(classes),
			softmax=False
		)

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

	def test_weights(self):
		classes = (np.arange(5) + 10).astype(np.float32)
		weights = np.array([0.3, 0.6, 0.9, 0.6, 0.3]).astype(np.float32)

		loss_fn = ProximalMaskedLoss(
			n=len(classes),
			softmax=False,
		)
		w_loss_fn = ProximalMaskedLoss(
			n=len(classes),
			softmax=False,
			weights=weights
		)

		y = torch.from_numpy(np.array([
			[0, 1, 0, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1]
		]).astype(np.float32))

		predictions = torch.from_numpy(np.array([
			[1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0]
		]).astype(np.float32))

		losses = [loss_fn(predictions[i:i+1], y[i:i+1]) for i in range(y.shape[0])]
		w_losses = [w_loss_fn(predictions[i:i+1], y[i:i+1]) for i in range(y.shape[0])]

		print(f"Losses: {losses}")
		print(f"Weighted Losses: {w_losses}")

		self.assertEqual(losses[0], losses[1])
		self.assertEqual(losses[0], losses[2])

		self.assertGreater(w_losses[1], w_losses[0])
		self.assertGreater(w_losses[1], w_losses[2])
		self.assertGreater(w_losses[0], w_losses[2])