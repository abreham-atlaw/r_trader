import typing
from abc import ABC

from core.agent.trader_action import TraderAction
from core.environment.trade_state import TradeState
from lib.rl.agent import ActionChoiceAgent
from core import Config


class ActionChoiceTrader(ActionChoiceAgent, ABC):

	def __init__(
			self,
			*args,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__trade_size_gap = trade_size_gap

	def _generate_actions(self, state: TradeState) -> typing.List[typing.Optional[TraderAction]]:
		pairs = state.get_market_state().get_tradable_pairs()

		amounts = [
			(i + 1) * self.__trade_size_gap
			for i in range(int(state.get_agent_state().get_margin_available() // self.__trade_size_gap))
		]

		actions: typing.List[typing.Optional[TraderAction]] = [
			TraderAction(
				pair[0],
				pair[1],
				action,
				margin_used=amount
			)
			for pair in pairs
			for action in [TraderAction.Action.BUY, TraderAction.Action.SELL]
			for amount in amounts
		]

		actions += [
			TraderAction(trade.get_trade().base_currency, trade.get_trade().quote_currency, TraderAction.Action.CLOSE)
			for trade in state.get_agent_state().get_open_trades()
		]

		actions.append(None)
		return actions
