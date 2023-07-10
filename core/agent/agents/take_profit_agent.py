import typing

from core import Config
from core.agent.trader_action import TraderAction
from core.environment.trade_state import TradeState, AgentState
from lib.rl.agent import Agent


class TakeProfitAgent(Agent):

	def __init__(self, *args, take_profit: float = Config.AGENT_TAKE_PROFIT, **kwargs):
		super().__init__(*args, **kwargs)
		self.__take_profit = take_profit

	def __should_take_profit(self, open_trade: AgentState.OpenTrade) -> bool:
		return open_trade.get_unrealized_profit() > self.__take_profit

	def _policy(self, state: TradeState) -> TraderAction:
		for open_trade in state.get_agent_state().get_open_trades():
			if self.__should_take_profit(open_trade):
				return TraderAction(
					open_trade.get_trade().base_currency,
					open_trade.get_trade().quote_currency,
					TraderAction.Action.CLOSE,
				)
		return super()._policy(state)
