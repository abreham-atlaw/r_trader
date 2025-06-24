import unittest

from core.utils.research.data.prepare.swg.swg.training import XGBoostSWGModel, SampleWeightGenerationModel
from .abstract_model_test import AbstractModelTest


class XGBoostModelTest(AbstractModelTest, unittest.TestCase):

	def _init_model(self) -> SampleWeightGenerationModel:
		return XGBoostSWGModel(norm=True)
