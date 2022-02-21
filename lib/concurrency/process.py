from typing import *

from core import Config
from multiprocessing import Process, Manager
import time
import os


SLEEP_LENGTH = 0.05
RETURN_KEY = "return_value"


class ConcurrentProcess(Process):

	def __init__(self, *args, function=None, function_arguments=None, return_storage=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.__function = function
		self.__function_arguments = function_arguments
		if function_arguments is None:
			self.__function_arguments = ()
		self.__return_storage = return_storage
		if return_storage is None:
			manager = Manager()
			self.__return_storage = manager.dict()
		self.__started = False

	def start(self):
		self.__started = True
		super().start()

	def run(self):
		if self.__function is not None:
			self.set_return(
				self.__function(*self.__function_arguments)
			)

	def get_return(self, join=True):
		if not self.has_started():
			self.start()
		if join:
			self.join()
		return self.__return_storage.get(RETURN_KEY)

	def set_return(self, value):
		self.__return_storage[RETURN_KEY] = value

	def has_started(self):
		return self.__started


class Pool(ConcurrentProcess):

	def __init__(self, max_process=Config.MAX_PROCESSES, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__processes: List[ConcurrentProcess] = None
		self.__max_process = max_process
		self.__map_function = None
		self.__map_args = None

	def map(self, function, args):
		self.__map_function = function
		self.__map_args = args

	def __check_and_wait(self):
		running_processes = [process for process in self.__processes if (process.has_started() and process.is_alive())]
		if len(running_processes) < self.__max_process:
			return
		time.sleep(SLEEP_LENGTH)
		self.__check_and_wait()

	def _create_process(self, function, args):
		return ConcurrentProcess(function=function, function_arguments=args)

	def run(self):
		self.__processes = [
			self._create_process(self.__map_function, args=args)
			for args in self.__map_args
		]
		for process in self.__processes:
			process.start()
			self.__check_and_wait()
		self.set_return(
			[process.get_return() for process in self.__processes]
		)

	@staticmethod
	def is_main_process() -> bool:
		return os.getpid() == Config.MAIN_PID
