import typing
from abc import ABC

from dataclasses import dataclass

from .action_choice_agent import ActionChoiceAgent
from ...utils.logger import Logger


class CumulativeRewardAgent(ActionChoiceAgent, ABC):

	@dataclass
	class RewardStore:
		initial_state: typing.Any
		action: typing.Any
		final_state: typing.Any
		value: float

	def __init__(
			self,
			*args,
			reward_cumulation_size: int = 5,
			reward_cumulation_discount: float = 1,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__size = reward_cumulation_size
		self.__discount = reward_cumulation_discount
		self.__queue: typing.List[CumulativeRewardAgent.RewardStore] = []

	def __add_to_queue(self, initial_state, action, final_state, value):
		for i, reward in enumerate(self.__queue[::-1]):
			reward.value += (self.__discount ** (i+1))*value
		self.__queue.append(CumulativeRewardAgent.RewardStore(initial_state, action, final_state, value))

	def _update_state_action_value(self, initial_state, action, final_state, reward):
		self.__add_to_queue(initial_state, action, final_state, reward)
		if len(self.__queue) > self.__size:
			Logger.info(f"CRA: Updating State Action Value(CUM={self.__size})")
			reward = self.__queue.pop(0)
			super()._update_state_action_value(
				reward.initial_state,
				reward.action,
				reward.final_state,
				reward.value
			)
