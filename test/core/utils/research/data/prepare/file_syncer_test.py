import unittest

import os
import random

from core.utils.research.data.prepare.file_syncer import FileSyncer


class FileSyncerTest(unittest.TestCase):

	def setUp(self):
		self.paths = [
			os.path.join("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/test/filesyncer", str(i))
			for i in range(2)
		]
		for path in self.paths:
			os.system(f"rm -rf {path}")
			os.mkdir(path)
			for i in range(10):
				if random.choice([True]*6 + [False]*4):
					os.system(f"touch {os.path.join(path, str(i))}")

		for path in self.paths:
			print(f"{path} Files:", '\n'.join(os.listdir(path)), sep="\n")

	def test_functionality(self):

		syncer = FileSyncer()

		syncer.sync(self.paths)
