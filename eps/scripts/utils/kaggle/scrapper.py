import math
import random
import time

from core.utils.kaggle.scrapper import ShareProcess, KaggleScraper


def share_notebooks():

	cookies_path = '/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/kaggle_cookies/abrehamatlaw0.json'
	notebook_urls = [
		f'https://www.kaggle.com/code/abrehamatlaw0/rtrader-ml-rsp-cnn-0-it-{i}-it-23-sw{j}/'
		for i in [23, 27]
		for j in [8]
		# for i in range(40)
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
		'abrehamalemu',
		'albertcamus0',
		'birukay',
		'nikolatesla0',
		'friedrichnietzche0'
	]

	scrapper_threads = []

	for i in range(math.ceil(len(notebook_urls) / threads)):
		scrapper_threads.append(ShareProcess(
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



def main():
	# remove_inputs()
	share_notebooks()
	# visit()
	# share_raw()


if __name__ == '__main__':
	main()
