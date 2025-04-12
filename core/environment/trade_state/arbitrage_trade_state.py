from typing import *

from dataclasses import dataclass



@dataclass
class ArbitradeTradeState:

	class MarginStage:
		STAGE_ZERO = 0
		STAGE_ONE = 1

	start_point: float
	checkpoints: Tuple[float, float]
	close_points: Tuple[float, float]
	instrument: Tuple[str, str]
	margin_stage: int = MarginStage.STAGE_ZERO

	def increment_margin_stage(self):
		if self.margin_stage == ArbitradeTradeState.MarginStage.STAGE_ONE:
			return
		self.margin_stage += 1

