from core import Config

if Config.USE_BINDINGS:
	from .trader_action import *

else:
	from ._trader_action import *
