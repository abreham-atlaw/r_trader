import unittest

from core.utils.kaggle.scrapper import KaggleScraper


class KaggleScrapperTest(unittest.TestCase):

	def setUp(self):
		self.scrapper = KaggleScraper(
			cookies_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies/inkoops.json"
		)

	def test_get_output_url(self):
		NOTEBOOK_URL = "https://www.kaggle.com/code/inkoops/rtrader-ml-runlive-sim-us-cnn-it-23-me-dk-stm/"
		VERSION = 130
		url = self.scrapper.get_output_url(NOTEBOOK_URL, VERSION)
		print(f"URL: {url}")
		self.assertTrue(url.startswith("https://www.kaggle.com/code/svzip/"))

