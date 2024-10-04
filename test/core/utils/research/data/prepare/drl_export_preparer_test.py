import unittest

from core.utils.research.data.prepare import DRLExportPreparer


class DRLExportPreparerTest(unittest.TestCase):

	def test_functionality(self):


		INPUT_DIRS = [
			"/home/abrehamatlaw/Downloads/Compressed/results_13/out"
		]

		INPUT_SIZE = 1157
		SEQ_LEN = 1033
		MA_WINDOW = 10
		OUT_DIR = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export"

		preparer = DRLExportPreparer(
			INPUT_SIZE,
			SEQ_LEN,
			MA_WINDOW,
			OUT_DIR
		)

		for input_dir in INPUT_DIRS:
			print(f"\n[+]Processing {input_dir}...")
			try:
				preparer.start(input_dir)
				print("[+]Done.")
			except Exception as ex:
				print(f"[-]Failed with {ex}")


