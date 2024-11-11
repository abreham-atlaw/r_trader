import unittest

from core.utils.research.data.prepare import BatchSizeModifier


class BatchSizeModifierTest(unittest.TestCase):

	def setUp(self):
		self.modifier = BatchSizeModifier(
			source_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train",
			target_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/batch_modified",
			target_batch_size=64
		)

	def test_functionality(self):
		self.modifier.start()
