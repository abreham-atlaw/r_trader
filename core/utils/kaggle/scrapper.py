import json
import os
import time
import typing

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class KaggleScraper:

	__KAGGLE_URL = "https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2F"

	def __init__(self, cookies_path: typing.Optional[str] = None):
		self.driver = self._configure_driver()
		if cookies_path is not None:
			self.load_cookies(cookies_path)

	@staticmethod
	def _configure_driver():
		return webdriver.Firefox()

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

	def __enable_edit(self):
		buttons = [button for button in self.driver.find_elements(By.TAG_NAME, value="button")]
		dropdown_button = [
			button for button in buttons
			if button.text == "Can view\narrow_drop_down"
		]
		if len(dropdown_button) == 0:
			return
		self._scroll_and_click(dropdown_button[0])

		elements = self.driver.find_elements(by=By.XPATH, value="//*[contains(text(), 'Can edit')]")
		clickable_elements = [element for element in elements if element.is_enabled() and element.is_displayed()]
		self._scroll_and_click(clickable_elements[0])
		time.sleep(2)

		if len(dropdown_button) > 1:
			self.__enable_edit()

	def share_notebook(self, notebook_url, username):
		self.driver.get(os.path.join(notebook_url, "settings"))
		self._scroll_down()
		username_input = WebDriverWait(self.driver, 20).until(
			EC.element_to_be_clickable((By.XPATH,  "//input[contains(@placeholder,'Search collaborators')]"))
		)
		username_input.send_keys(username)

		user_result = WebDriverWait(self.driver, 20).until(
			EC.element_to_be_clickable((By.XPATH, f"//*[contains(text(),'{username}')]"))
		)
		user_result.click()

		self.__enable_edit()

		share_action = self.driver.find_element(By.XPATH, "//*[contains(text(),'Save Changes')]")
		self._scroll_and_click(share_action)
		time.sleep(5)


# Usage
def main():
	cookies_path = '/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies/inkoops.json'
	notebook_urls = [
		f'https://www.kaggle.com/code/inkoops/rtrader-runlive-sim-cum-0-it-0-{i}'
		for i in range(3, 20)
	]
	usernames = [
		'bemnetatlaw',
		'abrehamatlaw0',
		'inkoops',
		'yosephmezemer',
		'napoleonbonaparte0',
		'biruk-ay',
		'albertcamus0'
	]

	scrapper = KaggleScraper(
		cookies_path=cookies_path
	)

	for i, notebook_url in enumerate(notebook_urls):
		for username in usernames:
			try:
				scrapper.share_notebook(notebook_url, username)
			except Exception as ex:
				print(f"Failed to share {notebook_url} to {username}. Reason: {ex}")
				pass
		print(f"Progress{(i+1)*100/len(notebook_urls): .2f}%...")


if __name__ == '__main__':
	main()
