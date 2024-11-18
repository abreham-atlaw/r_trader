import json
import math
import os
import random
import threading
import time
import typing
from multiprocessing import Process
from threading import Thread

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class KaggleScraper:

	__KAGGLE_URL = "https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2F"

	def __init__(self, cookies_path: typing.Optional[str] = None):
		self.driver = self._configure_driver()
		self.__cookies_path = cookies_path
		self.drivers = {}
		self.__initialized = False

	@staticmethod
	def _configure_driver():
		driver = webdriver.Firefox()
		driver.set_window_size(600, 800)
		driver.set_window_position(0, 1080)
		return driver

	def init(self):
		if self.__initialized:
			return
		if self.__cookies_path is not None:
			self.load_cookies(self.__cookies_path)
		self.__initialized = True

	def load_cookies(self, path):
		self.driver.get(self.__KAGGLE_URL)
		with open(path, 'r') as file:
			cookies = json.load(file)
			for cookie in cookies:
				self.driver.add_cookie(cookie)

	def _scroll_down(self):
		self.driver.execute_script("window.scrollTo(0, window.innerHeight);")

	@staticmethod
	def _scroll_to(element):
		element.location_once_scrolled_into_view

	def _scroll_and_click(self, element):
		self._scroll_to(element)
		element.click()

	def __add_user(self, username):
		self._scroll_down()

		username_input = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, "//input[contains(@placeholder,'Search collaborators')]"))
		)
		username_input.clear()
		time.sleep(5)
		username_input.send_keys(username)
		try:
			user_result = WebDriverWait(self.driver, 5).until(
				EC.element_to_be_clickable((By.XPATH, f"//*[contains(text(),'({username})')]"))
			)
			user_result.click()
		except TimeoutException:
			pass
		username_input.clear()

	def __enable_edit(self):
		divs = [div for div in self.driver.find_elements(By.TAG_NAME, value="div")]
		dropdown_button = [
			div for div in divs
			if div.text == "Can view"
		]
		if len(dropdown_button) == 0:
			return
		self._scroll_and_click(dropdown_button[0])

		elements = self.driver.find_elements(by=By.XPATH, value="//*[contains(text(), 'Can edit')]")
		clickable_elements = [element for element in elements if element.is_enabled() and element.is_displayed()]
		self._scroll_and_click(clickable_elements[-1])
		time.sleep(2)

		if len(dropdown_button) > 1:
			self.__enable_edit()

	def share_notebook(self, notebook_url, usernames, visit=False):
		self.driver.get(notebook_url)
		share_button = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, '//span[text()="Share"]'))
		)
		share_button.click()

		for username in usernames:
			self.__add_user(username)

		self.__enable_edit()
		share_action = self.driver.find_element(By.XPATH, "//*[contains(text(),'Save')]")
		self._scroll_and_click(share_action)
		time.sleep(5)
		if not visit:
			return
		for username in usernames:
			try:
				self.visit_as(username, notebook_url)
			except Exception as ex:
				print(f"Failed to visit {notebook_url} as {username}. Error:\n{ex}")

	def clear_inputs(self, notebook_url: str):
		self.driver.set_window_size(1920, 600)
		self.driver.get("https://www.kaggle.com/code/abrehamatlaw0/rtrader-datapreparer-cum-0-it-3/edit/run/196594046")
		time.sleep(30)
		notebooks_header = [div for div in self.driver.find_elements(By.TAG_NAME, "div") if "NOTEBOOKS" in div.text][-1]
		container: WebElement = notebooks_header.find_element(By.XPATH, "following-sibling::*")

		failed = False

		for element in container.find_elements(By.XPATH, "./*"):
			try:
				more_button = [button for button in element.find_elements(By.TAG_NAME, value="button") if "More actions for" in button.get_attribute("aria-label")][0]
				self._scroll_and_click(more_button)
				time.sleep(1)

				remove_button = [p for p in element.find_elements(By.TAG_NAME, value="p") if p.text == "Remove"][0]
				self._scroll_and_click(remove_button)
				time.sleep(1)
			except Exception as ex:
				failed = True

		if failed:
			self.clear_inputs(notebook_url)

	def visit(self, notebook_url: str, retries=10):
		while retries > 0:
			try:
				self.init()
				self.driver.get(os.path.join(notebook_url, "edit"))
				time.sleep(30)
				break
			except Exception as ex:
				print(f"Failed to visit {notebook_url}. Error:\n{ex}. Retries Left: {retries-1}")
				retries -= 1

	def visit_as(self, username: str, notebook_url: str):
		cookie_path = os.path.join(os.path.dirname(self.__cookies_path), f"{username}.json")
		if not os.path.exists(cookie_path):
			raise Exception(f"Can't find cookie for {username}")
		user_scrapper = KaggleScraper(cookie_path)
		user_scrapper.visit(notebook_url)
		user_scrapper.driver.close()


class ScrapperProcess(Process):

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


class VisitProcess(Process):

	def __init__(self, cookies_path: str, notebooks_url: typing.List[str]):
		super().__init__()
		self.__cookies_path = cookies_path
		self.__notebooks_url = notebooks_url
		self.__scrapper = None

	def run(self):
		scrapper = KaggleScraper(
			cookies_path=self.__cookies_path
		)

		for notebook_url in self.__notebooks_url:
			scrapper.visit(notebook_url)

		scrapper.driver.close()


def share_notebooks():

	cookies_path = '/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies/abrehamalemu.json'
	notebook_urls = [
		f'https://www.kaggle.com/code/abrehamalemu/rtrader-training-exp-0-cnn-{i}-cum-0-it-4-tot/'
		for i in range(177, 179)
	]
	random.shuffle(notebook_urls)
	threads = len(notebook_urls)
	# threads = 1
	usernames = [
		'bemnetatlaw',
		'abrehamatlaw0',
		'yosephmezemer',
		'napoleonbonaparte0',
		'inkoops',
		# 'abrehamalemu',
		'albertcamus0',
		'birukay',
		'nikolatesla0',
		'friedrichnietzche0'
	]

	scrapper_threads = []

	for i in range(math.ceil(len(notebook_urls) / threads)):
		scrapper_threads.append(ScrapperProcess(
			cookies_path=cookies_path,
			notebooks_url=notebook_urls[i*threads: (i+1)*threads],
			usernames=usernames
		))
		scrapper_threads[i].start()
		time.sleep(10)

	for i, scrapper_thread in enumerate(scrapper_threads):
		scrapper_thread.join()
		print(f"Progress{(i+1)*100/len(scrapper_threads): .2f}%...")


def share_raw():
	cookies_path = '/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies/abrehamalemu.json'
	notebook_urls = [
		f'https://www.kaggle.com/code/abrehamalemu/rtrader-training-exp-0-linear-{i}-cum-0-it-4-tot/'
		for i in range(122, 128)
	]
	random.shuffle(notebook_urls)
	threads = len(notebook_urls)
	# threads = 1
	usernames = [
		'bemnetatlaw',
		'abrehamatlaw0',
		'yosephmezemer',
		'napoleonbonaparte0',
		'inkoops',
		# 'abrehamalemu',
		'albertcamus0',
		'birukay',
		'nikolatesla0',
		'friedrichnietzche0'
	]

	scrapper = KaggleScraper(cookies_path)
	scrapper.init()
	for notebook in notebook_urls:
		scrapper.share_notebook(notebook, usernames=usernames)


def remove_inputs():
	cookies_path = '/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies/abrehamatlaw0.json'
	notebook_url = "https://www.kaggle.com/code/abrehamatlaw0/rtrader-datapreparer-cum-0-it-3/"
	scrapper = KaggleScraper(cookies_path)
	scrapper.init()
	scrapper.clear_inputs(notebook_url)


def visit():
	notebook_urls = [
		f'https://www.kaggle.com/code/abrehamatlaw0/rtrader-datapreparer-simsim-cum-0-it-2-{i}'
		for i in range(0, 4)
	]

	usernames = [
		'bemnetatlaw',
		'abrehamatlaw0',
		'napoleonbonaparte0',
		'inkoops',
		'abrehamalemu',
		'albertcamus0',
		'biruk-ay',
		'nikolatesla0',
		'friedrichnietzche0'
	]

	cookie_container = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies"

	processes = []

	for i, username in enumerate(usernames):
		print(f"Starting Visit Process for {username}...")
		cookies_path = os.path.join(cookie_container, f"{username}.json")
		process = VisitProcess(cookies_path=cookies_path, notebooks_url=notebook_urls)
		process.start()
		processes.append(process)
		print(f"Initialization {(i+1)*100/len(usernames):.2f}%...")
		time.sleep(10)
	for process in processes:
		process.join()


def main():
	# remove_inputs()
	share_notebooks()
	# visit()
	# share_raw()


if __name__ == '__main__':
	main()
