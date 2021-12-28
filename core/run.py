import sys

from core.agent.trader_agent import TraderAgent
from core.environment import TradeEnvironment, TrainingEnvironment, LiveEnvironment


def start_agent(environment: TradeEnvironment):
	agent = TraderAgent()
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


RUN_FUNCTIONS = {
	"live": run_live,
	"train": run_train
}

if __name__ == "__main__":

	mode = sys.argv[1]
	if mode not in RUN_FUNCTIONS.keys():
		raise Exception(f"Invalid Argument {mode}")

	RUN_FUNCTIONS[mode]()


