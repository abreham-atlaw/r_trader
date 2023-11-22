import unittest

import numpy as np
import torch

from core.utils.research.model.model.cnn.model import CNN


class CNNTest(unittest.TestCase):

	def test_functionality(self):

		model = torch.jit.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/1700508499.667843.pt")
		model.eval()

		DTYPE = torch.float32
		NP_DTYPE = np.float32

		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train/X/1700140166.854279.npy").astype(NP_DTYPE)[:]
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train/y/1700140166.854279.npy").astype(NP_DTYPE)[:]

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		y_hat_classes = np.argmax(y_hat.detach().numpy(), axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertEqual(y_hat_classes.shape, (X.shape[0],))
