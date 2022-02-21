from typing import *

import unittest
from unittest import mock

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.activations import relu, tanh

from lib.dnn.utils import KerasTrainer
from lib.dnn.utils import Optimizer


class OptimizerTest(unittest.TestCase):

	PARAMS = {
		"layers": [
			[60, 30, 10],
			[20, 40, 15],
		],
		"optimizer": [
			Adam(),
			SGD()
		],
		"loss": [
			mean_squared_error,
			mean_absolute_error
		],
		"activation": [
			relu,
			tanh
		]
	}

	OPTIMAL_PARAMS = {
		"layers": PARAMS["layers"][0],
		"optimizer": PARAMS["optimizer"][1],
		"loss": PARAMS["loss"][1],
		"activation": PARAMS["activation"][1]
	}

	class TestOptimizer(Optimizer):

		def __init__(self, mock_trainer):
			super().__init__()
			self.mock_trainer = mock_trainer

		def _generate_param_values(self) -> Dict:
			return OptimizerTest.PARAMS
		
		def _create_trainer(self, params: Dict) -> KerasTrainer:
			self.mock_trainer(**params)
			value = sum([
				sum(params.get("layers")),
				ord(params.get("optimizer").__class__.__name__[0]) % 100,
				len(params.get("loss").__name__) % 100,
				ord(params.get("activation").__name__[0]) % 100
			])
			print(f"Value: {value}\n\n")
			train_history = mock.Mock()
			train_history.history = {"loss": [value/(i*10000) for i in range(1, 6)]}
			self.mock_trainer.start.get_return = train_history, value / 10000
			return self.mock_trainer

	def test_optimize(self):
		mock_trainer = mock.Mock()
		optimizer = OptimizerTest.TestOptimizer(mock_trainer)
		optimal_params, min_loss = optimizer.optimize()
		
		for layers in OptimizerTest.PARAMS.get("layers"):
			for optimizer in OptimizerTest.PARAMS.get("optimizer"):
				for loss in OptimizerTest.PARAMS.get("loss"):
					for activation in OptimizerTest.PARAMS.get("activation"):
						mock_trainer.assert_any_call(
							layers=layers,
							optimizer=optimizer,
							loss=loss,
							activation=activation
						)
		
		self.assertEqual(optimal_params, OptimizerTest.OPTIMAL_PARAMS)


if __name__ == "__main__":
	unittest.main()
