import os
import sys


class RTraderApplication:

	class Mode:
		LIVE = 'live'
		TRAIN = 'train'

	def __init__(self, config=None):
		self.RUN_FUNCTIONS = {
			RTraderApplication.Mode.LIVE: self.__run_live,
			RTraderApplication.Mode.TRAIN: self.__run_train
		}
		self.config = config
		self.is_setup = False

	def __download_model(self, url:str, path: str):
		print(f"[+]Downloading Model from {url} to {path}...")
		os.system(f"wget '{url}' -O {path}")

	def __import_config(self):
		path = os.path.abspath(os.path.dirname(__file__))
		sys.path.append(path)
		import Config
		sys.path.remove(path)
		return Config

	def setup(self):
		print(f"[+]Setting up Application")
		if self.config is None:
			self.config = self.__import_config() 
		sys.setrecursionlimit(self.config.RECURSION_DEPTH)
		sys.path.append(self.config.BASE_DIR)
		if self.config.MODEL_DOWNLOAD:
			self.__download_model(self.config.MODEL_DOWNLOAD_URL, self.config.MODEL_PATH)
		self.is_setup = True

	def __start_agent(self, environment):
		from core.agent.trader_agent import TraderMonteCarloAgent

		agent = TraderMonteCarloAgent()
		agent.set_environment(environment)
		agent.loop()

	def __run_live(self):
		from core.environment import LiveEnvironment

		environment = LiveEnvironment()
		environment.start()
		self.__start_agent(environment)

	def __run_train(self):
		from core.environment import TrainingEnvironment

		environment = TrainingEnvironment()
		environment.start()
		self.__start_agent(environment)

	def run_args(self, args):
		mode = args[1]
		if mode not in self.RUN_FUNCTIONS.keys():
			raise Exception(f"Invalid Argument {mode}")

		self.run(mode)

	def run(self, mode: str):
		if not self.is_setup:
			self.setup()
		self.RUN_FUNCTIONS[mode]()
