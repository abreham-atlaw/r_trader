from core.utils.research.data.collect.analysis import ProfitPredictor


class ResearchProvider:

	@staticmethod
	def provide_profit_predictor() -> ProfitPredictor:
		return ProfitPredictor()
