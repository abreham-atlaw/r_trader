
class CurrencyNotFoundException(Exception):

	def __init__(self, currency):
		self.currency = currency

	def __str__(self):
		return "Currency not found: " + self.currency


class InsufficientFundsException(Exception):
	pass
