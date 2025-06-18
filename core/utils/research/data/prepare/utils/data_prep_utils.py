import typing

import numpy as np


class DataPrepUtils:

	@staticmethod
	def find_bound_index(bounds: typing.List[float], value: float) -> int:
		return np.sum(value >= np.array(bounds))
