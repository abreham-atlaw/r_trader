import unittest

from core.utils.research.data.prepare.swg.swg.training import LassoSWGModel, SampleWeightGenerationModel
from .abstract_model_test import AbstractModelTest


class LassoSWGModelTest(AbstractModelTest, unittest.TestCase):

	def _init_model(self) -> SampleWeightGenerationModel:
		return LassoSWGModel(norm=True)
