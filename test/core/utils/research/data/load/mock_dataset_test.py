import unittest

from core.utils.research.data.load.mock_dataset import MockDataset


class MockDatasetTest(unittest.TestCase):

	def setUp(self):

		self.SIZE = 1000
		self.SHAPES = [(1148,), (432,)]
		self.dataset = MockDataset(
			size=self.SIZE,
			shapes=self.SHAPES
		)

	def test_functionality(self):
		size = len(self.dataset)
		self.assertEqual(size, self.SIZE)

		X, y = self.dataset[1]
		self.assertEqual(X.shape, self.SHAPES[0])
		self.assertEqual(y.shape, self.SHAPES[1])
