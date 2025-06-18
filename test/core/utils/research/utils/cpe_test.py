import unittest

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.eval.mlpl_evaluator.losses.unbatched_pml_loss import UnbatchedProximalMaskedLoss
from core.utils.research.losses import ClassPerformanceLoss
from core.utils.research.utils.cpe import ClassPerformanceEvaluator
from lib.utils.torch_utils.model_handler import ModelHandler


class ClassPerformanceEvaluatorTest(unittest.TestCase):

	def test_functionality(self):
		dataset = BaseDataset(
			root_dirs=[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			]
		)

		MODEL_PATH = "/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-45-cum-0-it-29-tot.zip"

		dataloader = DataLoader(
			dataset,
			batch_size=8,
			shuffle=False
		)
		classes = len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1

		evaluator = ClassPerformanceEvaluator(
			loss=ClassPerformanceLoss(
				loss_fn=UnbatchedProximalMaskedLoss(
					n=classes,
					softmax=True
				),
				n=classes,
				nan_to=0
			),
			dataloader=dataloader,
			model_class_range=(0, classes)
		)

		model = ModelHandler.load(MODEL_PATH)

		loss = evaluator.evaluate(model)

		print(loss)

		self.assertEqual(loss.shape[0], classes)

		plt.plot(loss.numpy())
		plt.show()
