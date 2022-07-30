from typing import *

from lib.concurrency.mc.worker import MonteCarloWorkerPool, MonteCarloWorkerAgent
from .mcw import TraderMonteCarloWorkerAgent
from core import Config


class TraderMonteCarloWorkerPool(MonteCarloWorkerPool):

	def __init__(self, prepare_agent: Callable=None, processes=Config.MC_WORKERS):
		super(TraderMonteCarloWorkerPool, self).__init__(processes=processes)
		self.__prepare_agent = prepare_agent

	def _create_worker(self) -> MonteCarloWorkerAgent:
		agent = TraderMonteCarloWorkerAgent()
		if self.__prepare_agent is not None:
			self.__prepare_agent(agent)

		return agent
