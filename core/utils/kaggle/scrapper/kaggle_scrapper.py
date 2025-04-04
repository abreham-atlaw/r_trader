import json
import os
import sqlite3
import time
import typing

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class KaggleScraper:

	__KAGGLE_URL = "https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2F"

	def __init__(
			self,
			cookies_path: typing.Optional[str] = None,
			tmp_path: str = "/tmp/"
	):
		self.driver = self._configure_driver()
		self.__cookies_path = cookies_path
		self.drivers = {}
		self.__initialized = False
		self.__tmp_path = tmp_path

	@staticmethod
	def _configure_driver():
		options = webdriver.FirefoxOptions()

		options.set_preference("browser.download.panel.shown", True)
		options.set_preference("browser.download.manager.retention", 2)  # Keep downloads in history
		options.set_preference("places.history.enabled", True)
		options.set_preference("browser.history.updateDelayMs", 0)
		options.set_preference("places.frecency.numVisits", 1)  # Prioritize recent visits
		options.set_preference("places.frecency.updateIdle", False)  # Prevent idle-based delay

		driver = webdriver.Firefox(options=options)
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
			EC.element_to_be_clickable((By.XPATH, "//input[contains(@placeholder,'Share with people or groups')]"))
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

	def __open_notebook_version(self, notebook_url: str, version: int = None):
		self.driver.get(notebook_url)
		if version is None:
			return
		version_selection_button = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, f"//button[contains(normalize-space(.), 'Version ')]"))
		)
		# version_selection_button = next(filter(lambda b: "Version " in b.text, self.driver.find_elements(By.TAG_NAME, value="button")))
		version_selection_button.click()

		more_button = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, f"//div[@title='Version {version}']/../.."))
		)

		more_button = WebDriverWait(
			WebDriverWait(self.driver, 5).until(
				EC.element_to_be_clickable((By.XPATH, f"//div[@title='Version {version}']/../.."))
			),
			5
		).until(
			EC.element_to_be_clickable((By.XPATH, ".//button[@title='More options for this version']"))
		)
		self._scroll_and_click(more_button)

		view_version_button = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, "//p[text()='View full version']"))
		)
		self._scroll_and_click(view_version_button)

	def __open_notebook_output_tab(self):
		if "?scriptVersionId" in self.driver.current_url:
			url = self.driver.current_url.replace("?scriptVersionId", "/output?scriptVersionId")
		else:
			url = self.driver.current_url + "/output"
		self.driver.get(url)

	def __download_output(self):
		more_button = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, "//button[@title='Output actions']"))
		)
		self._scroll_and_click(more_button)
		download_button = WebDriverWait(self.driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, "//p[text() = 'Download output']"))
		)
		self._scroll_and_click(download_button)

	def __cancel_download(self):
		self.driver.get("about:downloads")
		self.driver.find_elements(By.CLASS_NAME, "downloadButton")[0].click()

	def __load_driver_db(self):
		db_path = os.path.join(self.driver.capabilities["moz:profile"], "places.sqlite")
		clone_path = os.path.join(self.__tmp_path, f"{time.time()}-places.sqlite")
		os.system(f"cp '{db_path}' '{clone_path}'")
		conn = sqlite3.connect(clone_path)
		return conn

	def __sync_db(self):
		self.driver.execute_script("""
			(async () => {
				let { PlacesUtils } = ChromeUtils.import("resource://gre/modules/PlacesUtils.jsm");
				await PlacesUtils.history.flush();  // Ensures pending writes are committed
			})();
		""")

	def __get_visit_history(self) -> typing.List[str]:
		self.__sync_db()
		conn = self.__load_driver_db()
		cursor = conn.cursor()
		cursor.execute("SELECT * FROM moz_places;")
		results = cursor.fetchall()
		return list(map(
			lambda r: r[1],
			results
		))

	def get_output_url(self, notebook_url: str, version: int = None) -> str:
		self.init()
		self.driver.set_window_size(1200, 800)
		self.__open_notebook_version(notebook_url, version)
		self.__open_notebook_output_tab()
		self.__download_output()
		self.__cancel_download()
		time.sleep(60)
		url = self.__get_visit_history()[-1]
		return url
