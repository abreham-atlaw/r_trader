import unittest

import numpy as np
import torch

from core.utils.research.losses import OutputBatchVarianceLoss


class OutputBatchVarianceLossTest(unittest.TestCase):

	def test_functionality(self):

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

		score_fn = OutputBatchVarianceLoss(softmax=True)

		score = [
			score_fn(batches[i])
			for i in range(len(batches))
		]

		self.assertGreater(score[1], score[0])
		self.assertGreater(score[2], score[1])

