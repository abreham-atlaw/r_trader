import unittest

import torch
import numpy as np

from core.utils.research.losses import OutputBatchClassVariance


class OutputBatchClassVarianceTest(unittest.TestCase):

	def test_functionality(self):
		classes = (np.arange(3) + 10).astype(np.float32)

		batches = [
			torch.from_numpy(np.array([
				[0.2, 0.3, 0.5],
				[0.1, 0.8, 0.1],
				[1, 0, 0]
			]).astype(np.float32)),

			torch.from_numpy(np.array([
				[0.2, 0.3, 0.5],
				[0.26, 0.27, 0.47],
				[0.14, 0.33, 0.53],
			]).astype(np.float32)),

			torch.from_numpy(np.array([
				[0.2, 0.3, 0.5],
				[0.2, 0.3, 0.5],
				[0.2, 0.3, 0.5],
			]).astype(np.float32)),

		]

		score_fn = OutputBatchClassVariance(softmax=True, classes=classes)

		score = [
			score_fn(batches[i], None)
			for i in range(len(batches))
		]

		self.assertLess(score[1], score[0])
		self.assertLess(score[2], score[1])

