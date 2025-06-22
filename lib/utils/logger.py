from typing import *

from datetime import datetime
import os

from core.Config import LOGGING, LOGGING_PID, LOGGING_CONSOLE, LOGGING_FILE_PATH


class Logger:
	class Colors:
		PURPLE = '\033[95m'
		OKBLUE = '\033[94m'
		OKCYAN = '\033[96m'
		OKGREEN = '\033[92m'
		WARNING = '\033[30;103m'
		FAIL = '\033[97;101m'
		ENDC = '\033[0m'
		BOLD = '\033[1m'
		UNDERLINE = '\033[4m'

	@staticmethod
	def print(*args, color: Union[str, None] = None, prefix: Union[str, None] = None, **kwargs):
		if not LOGGING:
			return
		if color is None:
			color = Logger.Colors.ENDC
		if prefix is None:
			prefix = ""
		prefix = f"[{datetime.now()}] {prefix}"
		if LOGGING_PID:
			prefix = f"PID:{os.getpid()} {prefix}"
		if not LOGGING_CONSOLE:
			kwargs["file"] = open(LOGGING_FILE_PATH, "a")
		print(color, prefix, *args, Logger.Colors.ENDC, **kwargs)
		if not LOGGING_CONSOLE:
			kwargs["file"].close()



	@staticmethod
	def log_function(func, args, kwargs, prefix=None):
		Logger.info(f"Starting {func.__name__} with args={args}, kwargs={kwargs} ...", prefix=prefix)
		return_value = func(*args, **kwargs)
		Logger.info(f"Done {func.__name__} with args={args}, kwargs={kwargs} => returned {return_value}.", prefix=prefix)
		return return_value

	@staticmethod
	def logged_method(prefix=None):

		def decorator(func):

			def wrapper(*args, **kwargs):
				return Logger.log_function(func, args, kwargs, prefix=prefix)
			return wrapper

		return decorator

	@staticmethod
	def logged_method(func):

		def wrapper(self, *args, **kwargs):
			args = (self,) + args
			return Logger.log_function(func, args, kwargs, prefix=f"{self.__class__.__name__}:")
		return wrapper

	@staticmethod
	def info(*args: Any, **kwargs: Any):
		Logger.print(*args, color=Logger.Colors.OKBLUE, **kwargs)

	@staticmethod
	def warning(*args: Any, **kwargs: Any):
		Logger.print(*args, Logger.Colors.WARNING, **kwargs)
	
	@staticmethod
	def error(*args: Any, **kwargs: Any):
		Logger.print(*args, Logger.Colors.FAIL, **kwargs)

	@staticmethod
	def success(*args: Any, **kwargs: Any):
		Logger.print(*args, Logger.Colors.OKGREEN, **kwargs)
