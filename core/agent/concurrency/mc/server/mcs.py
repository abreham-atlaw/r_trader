
from core.agent.agents import TraderMonteCarloAgent, TraderAgent

from lib.concurrency.mc.server import MonteCarloServer, MonteCarloServerAgent
from lib.network.rest_interface import Serializer
from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer
from core import Config


class TraderMonteCarloServerSimulator(MonteCarloServerAgent, TraderAgent):
	pass


class TraderMonteCarloServer(MonteCarloServer):

	def __init__(self, agent):
		self.__agent = agent
		super().__init__(port=Config.MC_SERVER_PORT)

	def _init_agent(self) -> MonteCarloServerAgent:
		return self.__agent

	def _init_graph_serializer(self) -> Serializer:
		return TraderNodeSerializer()
