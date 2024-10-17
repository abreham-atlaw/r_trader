import torch
import numpy as np

import unittest

from core.utils.research.losses import PredictionConfidenceScoreLoss


class PredictionConfidenceScoreLossTest(unittest.TestCase):

	def test_functionality(self):
		predictions = torch.from_numpy(np.array([
			[0.2, 0.3, 0.5],
			[0.1, 0.8, 0.1],
			[1, 0, 0]
		]).astype(np.float32))

		score_fn = PredictionConfidenceScoreLoss(softmax=True)

		score = [
			score_fn(predictions[i: i + 1])
			for i in range(len(predictions))
		]

		self.assertLess(score[1], score[0])
		self.assertLess(score[2], score[1])
