from typing import *

from dataclasses import dataclass


@dataclass
class TraderAction:
	class Action:
		BUY = 1
		CLOSE = 2
		SELL = 0

	base_currency: str
	quote_currency: str
	action: int
	margin_used: float = None
	units: int = None

	def __eq__(self, other):
		if not isinstance(other, TraderAction):
			return False
		return \
			self.base_currency == other.base_currency and \
			self.quote_currency == other.quote_currency and \
			self.action == other.action and \
			(
				self.margin_used == other.margin_used or
				self.units == other.units
			)
