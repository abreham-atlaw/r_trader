import unittest

import numpy as np
import torch

from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer


class CNNTest(unittest.TestCase):

	def test_functionality(self):
		VOCAB_SIZE = 449

		model = CNN(
			VOCAB_SIZE,
			conv_channels=[1, 64, 128], 
			kernel_sizes=[3, 3]
		)
		model.load_state_dict(
			torch.load(
				'/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/actual_cnn_model.pth',
				map_location=torch.device('cpu')
			)
		)
		model.eval()

		DTYPE = torch.float32
		NP_DTYPE = np.float32

		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train/X/1699605237.250606.npy").astype(NP_DTYPE)[:]
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train/y/1699605237.250606.npy").astype(NP_DTYPE)[:]

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		y_hat_classes = np.argmax(y_hat.detach().numpy(), axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertEqual(y_hat_classes.shape, (X.shape[0],))
