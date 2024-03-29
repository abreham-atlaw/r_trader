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
	def _configure_driver(self):
		return webdriver.Firefox()

	def load_cookies(self, path):
		self.driver.get(self.__KAGGLE_URL)
		with open(path, 'r') as file:
			cookies = json.load(file)
			for cookie in cookies:
				self.driver.add_cookie(cookie)

	def _scroll_down(self):
		self.driver.execute_script("window.scrollTo(0, window.innerHeight);")

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

		share_action = self.driver.find_element(By.XPATH, "//*[contains(text(),'Save Changes')]")
		share_action.click()
		time.sleep(5)



# Usage
def main():
	cookies_path = '/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/cookies.json'
	notebook_urls = [
		f'https://www.kaggle.com/code/abreham/chess-transitionprobability-collector-1-{i}'
		for i in range(20, 49)
	]
	usernames = [
		'abrehama',
		'abrehamatlaw',
	]

	scraper = KaggleScraper(cookies_path)

	for i, notebook_url in enumerate(notebook_urls):
		for username in usernames:
			try:
				scraper.share_notebook(notebook_url, username)
			except:
				pass
		print(f"Progress{(i+1)*100/len(notebook_urls): .2f}%...")


if __name__ == '__main__':
	main()
