import time
import typing
from multiprocessing import Process

from .kaggle_scrapper import KaggleScraper


class ShareProcess(Process):

	def __init__(self, cookies_path: str, notebooks_url: typing.List[str], usernames: typing.List[str]):
		super().__init__()
		self.__cookies_path = cookies_path
		self.__notebooks_url = notebooks_url
		self.__usernames = usernames
		self.__scrapper = None

	def run_scrapper(self, scrapper: KaggleScraper):
		successful_notebooks = []
		for notebook in self.__notebooks_url:
			try:
				scrapper.share_notebook(notebook, self.__usernames)
				print(f"Shared {notebook}...")
				successful_notebooks.append(notebook)

			except Exception as ex:
				print(f"Failed to Share {notebook}.")

		return successful_notebooks

	def init_scrapper(self) -> KaggleScraper:
		print(f"Initializing scrapper")
		try:
			if self.__scrapper is None:
				self.__scrapper = KaggleScraper(
					cookies_path=self.__cookies_path
				)
			self.__scrapper.init()
			return self.__scrapper

		except Exception:
			print("Failed to initialize scrapper. Retrying...")
			time.sleep(5)
			return self.init_scrapper()

	def run(self):
		scrapper = self.init_scrapper()
		successful = self.run_scrapper(scrapper)
		self.__notebooks_url = [notebook for notebook in self.__notebooks_url if notebook not in successful]
		if len(self.__notebooks_url) > 0:
			print(f"Found {len(self.__notebooks_url)} failed notebooks. Retrying...")
			return self.run()
		scrapper.driver.close()

