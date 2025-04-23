import unittest

import torch

from core.utils.research.eval.mlpl_evaluator.losses.unbatched_pml_loss import UnbatchedProximalMaskedLoss
from core.utils.research.losses.class_performance_loss import ClassPerformanceLoss


class ClassPerformanceLossTest(unittest.TestCase):

	def test_functionality(self):

		classes = (torch.arange(5) + 10)

		loss_fn = ClassPerformanceLoss(
			loss_fn=UnbatchedProximalMaskedLoss(
				n=len(classes),
				softmax=False
			),
			n=len(classes),
			nan_to=-1
		)

		y = torch.Tensor([
			[0, 0, 1, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0],
			[1, 0, 0, 0, 0]
		])

		y_hat = torch.Tensor([
			[0, 0, 1, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0.5, 0.5, 0, 0],
			[0.2, 0.8, 0, 0, 0]
		])

		loss = loss_fn(y_hat, y)
		print(loss)

		self.assertEqual(loss.shape[0], len(classes))

