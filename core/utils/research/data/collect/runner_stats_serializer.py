from typing import Dict

from core.utils.research.data.collect.runner_stats import RunnerStats
from lib.network.rest_interface import Serializer


class RunnerStatsSerializer(Serializer):

	def __init__(self):
		super().__init__(RunnerStats)

	def serialize(self, data: RunnerStats) -> Dict:
		return data.__dict__.copy()

	def deserialize(self, json_: Dict) -> RunnerStats:
		if json_.get("_id"):
			json_.pop("_id")

		if json_.get("model_loss"):
			json_["model_losses"] = (json_.pop("model_loss"), 0.0)

		return RunnerStats(**json_)
