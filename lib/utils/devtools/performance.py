import json
from datetime import datetime

import os

import numpy as np

from core.Config import BASE_DIR

durations = {

}

possible_state_visits = []
valid_actions = []
prediction_inputs = []

stat_dump_path = os.path.join(BASE_DIR, f"{datetime.now().timestamp()}.json")


def track_performance(key, func):
	start_time = datetime.now()
	rv = func()
	durations[key] = durations.get(key, []) + [(datetime.now() - start_time).total_seconds()]

	with open(stat_dump_path, "w") as f:
		json.dump({
			"durations": {
				key: {
					"total": sum(durations[key]),
					"avg": float(np.mean(durations[key])),
					"iterations": len(durations[key]),
					"values": durations[key]
				}
				for key in durations
			},

		}, f)

	return rv


def track_func_performance(key=None):
	def decorator(func):
		def wrapper(*args, **kwargs):
			func_key = key
			if key is None:
				func_key = func.__name__
			return track_performance(func_key, lambda: func(*args, **kwargs))
		return wrapper
	return decorator
