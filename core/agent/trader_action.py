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
