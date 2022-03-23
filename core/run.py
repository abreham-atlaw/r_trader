import sys


def start_agent(environment):
	agent = TraderMonteCarloAgent()
	agent.set_environment(environment)
	agent.loop()


def run_live():
	environment = LiveEnvironment()
	environment.start()
	start_agent(environment)


def run_train():
	environment = TrainingEnvironment()
	environment.start()
	start_agent(environment)


def setup():
	import Config
	sys.setrecursionlimit(Config.RECURSION_DEPTH)
	sys.path.append(Config.BASE_DIR)


RUN_FUNCTIONS = {
	"live": run_live,
	"train": run_train
}

if __name__ == "__main__":

	mode = sys.argv[1]
	if mode not in RUN_FUNCTIONS.keys():
		raise Exception(f"Invalid Argument {mode}")

	setup()
	from core.agent.trader_agent import TraderMonteCarloAgent
	from core.environment import LiveEnvironment
	try:
		from core.environment import TrainingEnvironment
	except ImportError:
		print("Couldn't Import TrainingEnvironment")
	RUN_FUNCTIONS[mode]()
