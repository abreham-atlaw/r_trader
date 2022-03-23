import os
import sys



class RTraderApplication:

	def __init__(self):
		self.RUN_FUNCTIONS = {
			"live": self.__run_live,
			"train": self.__run_train
		}

	def __download_model(self, url:str, path: str):
		print(f"[+]Downloading Model from {url} to {path}...")
		os.system(f"wget '{url}' -O {path}")

	def __setup(self):
		import Config
		sys.setrecursionlimit(Config.RECURSION_DEPTH)
		sys.path.append(Config.BASE_DIR)
		if Config.MODEL_DOWNLOAD:
			self.__download_model(Config.MODEL_DOWNLOAD_URL, Config.MODEL_PATH)

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

	def run(self, mode):
		self.__setup()
		self.RUN_FUNCTIONS[mode]()


if __name__ == "__main__":

	app = RTraderApplication()
	app.run(sys.argv)
