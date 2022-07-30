from typing import *

import threading

from lib.utils.logger import Logger


class LockManager:

	def __init__(self):
		self.__locks: List[Tuple[object, threading.Lock]] = []

	def __get_lock(self, var, create=False) -> Union[threading.Lock, None]:
		for v, lock in self.__locks:
			if v is var:
				return lock
		if not create:
			return None
		lock = threading.Lock()
		self.__locks.append((var, lock))
		return lock

	def lock(self, var):
		self.__get_lock(var, create=True).acquire()

	def unlock(self, var):
		self.__get_lock(var).release()

	def lock_and_do(self, var, func):
		self.lock(var)
		try:
			return func()
		finally:
			self.unlock(var)

