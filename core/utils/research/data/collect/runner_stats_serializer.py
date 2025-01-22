from typing import Dict

from core import Config
from core.utils.research.data.collect.runner_stats import RunnerStats
from lib.network.rest_interface import Serializer


class RunnerStatsSerializer(Serializer):

	def __init__(self):
		super().__init__(RunnerStats)

	def serialize(self, data: RunnerStats) -> Dict:
		json_ = data.__dict__.copy()
		if "profit" in json_.keys():
			json_.pop("profit")
		if "real_profit" in json_.keys():
			json_.pop("real_profit")
		return json_

	def deserialize(self, json_: Dict) -> RunnerStats:
		if json_.get("_id"):
			json_.pop("_id")

		if json_.get("branch"):
			json_.pop("branch")

		if "real_profits" in json_.keys():
			json_.pop("real_profits")

		if "profits" not in json_.keys():
			json_["profits"] = [json_.pop("profit")] + [0 for _ in range(len(json_["session_timestamps"]) - 1)]

		if "profit" in json_.keys():
			json_.pop("profit")

		if "model_losses_map" not in json_.keys():
			json_["model_losses_map"] = {Config.RunnerStatsLossesBranches.main: tuple(json_["model_losses"])}

		if "model_losses" in json_.keys():
			json_.pop("model_losses")

		return RunnerStats(**json_)
