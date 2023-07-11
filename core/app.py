import os
import sys


class RTraderApplication:

	class Mode:
		LIVE = 'live'
		TRAIN = 'train'
		MC_SERVER = "mc-server"
		MC_QUEEN = "mc-queen"
		MC_WORKER = "mc-worker"
		MC_WORKER_POOL = "mc-worker-pool"
		ARBITRAGE = "arbitrage"
		TREND_TAKE_PROFIT = "trend-takeprofit"

	def __init__(self, config=None):
		self.RUN_FUNCTIONS = {
			RTraderApplication.Mode.LIVE: self.__run_live,
			RTraderApplication.Mode.TRAIN: self.__run_train,
			RTraderApplication.Mode.MC_SERVER: self.__run_mc_server,
			RTraderApplication.Mode.MC_QUEEN: self.__run_mc_queen,
			RTraderApplication.Mode.MC_WORKER: self.__run_mc_worker,
			RTraderApplication.Mode.MC_WORKER_POOL: self.__run_mc_worker_pool,
			RTraderApplication.Mode.ARBITRAGE: self.__run_arbitrage,
			RTraderApplication.Mode.TREND_TAKE_PROFIT: self.__run_trend_takeprofit
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

	def __init_db(self):
		from lib.db import initialize_connection
		initialize_connection(
			db_host=self.config.DEFAULT_PG_CONFIG.get("host"),
			db_port=self.config.DEFAULT_PG_CONFIG.get("port"),
			db_user=self.config.DEFAULT_PG_CONFIG.get("user"),
			db_password=self.config.DEFAULT_PG_CONFIG.get("password"),
			db_name=self.config.DEFAULT_PG_CONFIG.get("database"),
		)

	def setup(self):
		print(f"[+]Setting up Application")
		if self.config is None:
			self.config = self.__import_config()
		sys.setrecursionlimit(self.config.RECURSION_DEPTH)
		sys.path.append(self.config.BASE_DIR)

		for model_config in self.config.PREDICTION_MODELS:
			if model_config.download:
				self.__download_model(model_config.url, model_config.path)

		self.is_setup = True

		try:
			self.__init_db()
		except Exception as ex:
			print(f"Couldn't Initialize DB.\n %s" % (ex,))

	def __start_agent(self, environment):
		from core.agent.agents import TraderAgent

		agent = TraderAgent()
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

	@staticmethod
	def __setup_agent(agent):
		from core.environment import LiveEnvironment
		print("Setting up Environment")
		environment = LiveEnvironment()
		environment.start()
		agent.set_environment(environment)
		print("Environment Set Up")

	def __run_mc_server(self):
		print("Running MC-Server")
		from core.agent.concurrency.mc.server import TraderMonteCarloServer, TraderMonteCarloServerSimulator

		agent = TraderMonteCarloServerSimulator()
		self.__setup_agent(agent)

		server = TraderMonteCarloServer(agent)
		server.start()

	def __run_mc_worker(self):
		print("Running MC-Worker")
		from core.agent.concurrency.mc.worker import TraderMonteCarloWorkerAgent

		worker = TraderMonteCarloWorkerAgent()
		self.__setup_agent(worker)
		worker.start()

	def __run_mc_worker_pool(self):
		print("Running MC-Worker-Pool")
		from core.agent.concurrency.mc.worker import TraderMonteCarloWorkerPool

		pool = TraderMonteCarloWorkerPool(prepare_agent=self.__setup_agent)
		pool.start()

	def __run_mc_queen(self):
		print("Running MC-Queen")
		from core.agent.concurrency.mc.queen import TraderMonteCarloQueen

		queen = TraderMonteCarloQueen()
		self.__setup_agent(queen)
		queen.loop()

	def __run_arbitrage(self):
		print("Running Arbitrage Trader...")
		from core.agent.agents import ArbitrageTraderAgent

		agent = ArbitrageTraderAgent()
		self.__setup_agent(agent)
		agent.loop()

	def __run_trend_takeprofit(self):
		print("Running Trend Takeprofit Trader...")
		from core.agent.agents import TrendTakeProfitAgent

		agent = TrendTakeProfitAgent()
		self.__setup_agent(agent)
		agent.loop()

	def run_args(self, args):
		mode = args[1]
		if mode not in self.RUN_FUNCTIONS.keys():
			raise Exception(f"Invalid Argument {mode}")

		self.run(mode)

	def run(self, mode: str):
		if not self.is_setup:
			self.setup()
		self.RUN_FUNCTIONS[mode]()
