from lib.concurrency.mc.worker import MonteCarloWorkerAgent
from lib.network.rest_interface import Serializer
from core.agent.agents import TraderMonteCarloAgent
from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer, TradeStateSerializer
from core import Config


class TraderMonteCarloWorkerAgent(TraderMonteCarloAgent, MonteCarloWorkerAgent):

	def __init__(self, *args, step_time=Config.MC_WORKER_STEP_TIME, **kwargs):
		super().__init__(server_url=Config.MC_SERVER_URL, step_time=step_time)

	def _init_state_serializer(self) -> Serializer:
		return TradeStateSerializer()

	def _init_graph_serializer(self) -> Serializer:
		return TraderNodeSerializer()
