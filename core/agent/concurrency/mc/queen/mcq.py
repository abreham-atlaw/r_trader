

from lib.concurrency.mc.queen import MonteCarloQueen
from lib.network.rest_interface import Serializer
from core.agent.trader_agent import TraderMonteCarloAgent
from core.agent.concurrency.mc.data.serializer import TraderActionSerializer, TradeStateSerializer
from core import Config


class TraderMonteCarloQueen(MonteCarloQueen, TraderMonteCarloAgent):

	def __init__(self, *args, server_url=Config.MC_SERVER_URL,**kwargs):
		super().__init__(server_url=server_url, *args, **kwargs)

	def _init_state_serializer(self) -> Serializer:
		return TradeStateSerializer()

	def _init_action_serializer(self) -> Serializer:
		return TraderActionSerializer()
