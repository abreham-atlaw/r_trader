import typing
from abc import ABC

from core.environment.trade_state import TradeState
from core.agent.trader_action import TraderAction
from core import Config

from .action_choice_trader import ActionChoiceTrader
from .action_recommendation_agent import ActionRecommendationTrader


class ActionChoiceBalancerTrader(ActionChoiceTrader, ActionRecommendationTrader, ABC):

	def __init__(
			self,
			*args,
			num_actions: float = Config.AGENT_NUM_ACTIONS,
			recommendation_percent: float = Config.AGENT_RECOMMENDATION_PERCENT,
			**kwargs
	):
		super().__init__(*args, num_actions=int(num_actions*recommendation_percent))

	def __generate_static_actions(self, state: TradeState) -> typing.List[TraderAction]:
		return ActionChoiceTrader._generate_actions(self, state)

	def __generate_recommended_actions(self, state: TradeState) -> typing.List[TraderAction]:
		return ActionRecommendationTrader._generate_actions(self, state)

	def __select_actions(
			self,
			static: typing.List[TraderAction],
			recommended: typing.List[TraderAction]
	) -> typing.List[TraderAction]:
		return recommended + static[:self._num_actions - len(recommended)]

	def _generate_actions(self, state: TradeState) -> typing.List[typing.Optional[TraderAction]]:
		return self.__select_actions(
			self.__generate_static_actions(state),
			self.__generate_recommended_actions(state)
		)
