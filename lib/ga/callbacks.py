from typing import *

from .species import Species


class Callback:

	def __init__(self):
		pass

	def on_epoch_end(self, population: List[Species]):
		pass

	def on_epoch_start(self, population: List[Species]):
		pass
