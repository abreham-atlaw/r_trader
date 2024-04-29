from core import Config
from lib.rl.agent.cra import CumulativeRewardAgent


class CumulativeRewardTraderAgent(CumulativeRewardAgent):

	def __init__(
			self,
			*args,
			reward_cumulation_size: int = Config.AGENT_CRA_SIZE,
			reward_cumulation_discount: float = Config.AGENT_CRA_DISCOUNT,
			**kwargs
	):
		super().__init__(
			*args,
			reward_cumulation_size=reward_cumulation_size,
			reward_cumulation_discount=reward_cumulation_discount,
			**kwargs
		)
