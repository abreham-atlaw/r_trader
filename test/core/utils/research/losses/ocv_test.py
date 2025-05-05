import unittest

import numpy as np
import torch

from core.utils.research.losses import OutputClassesVarianceScore


class OutputClassesVarianceTest(unittest.TestCase):

	def test_functionality(self):

		predictions = torch.from_numpy(np.array([
			[0.2, 0.3, 0.5],
			[0.1, 0.8, 0.1],
			[1, 0, 0]
		]).astype(np.float32))

		score_fn = OutputClassesVarianceScore(softmax=True)

		score = [
			score_fn(predictions[i: i+1])
			for i in range(len(predictions))
		]

		self.assertGreater(score[1], score[0])
		self.assertGreater(score[2], score[1])

