from requests.exceptions import HTTPError

from core import Config
from lib.utils.logger import Logger


def network_call(func):

	def wrapper(*args, **kwargs):
		tries = Config.NETWORK_TRIES
		while tries is None or tries > 0:
			try:
				return func(*args, **kwargs)
			except (HTTPError, ConnectionError):
				Logger.warning(f"Network Call {func.__name__} Failed.")
				if tries is not None:
					tries -= 1
					Logger.warning(f"Tries Left {tries}")

		raise HTTPError

	return wrapper
