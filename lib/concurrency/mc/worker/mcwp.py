from abc import ABC, abstractmethod

from multiprocessing import Process
import time

from .mcw import MonteCarloWorkerAgent


class MonteCarloWorkerPool(ABC):

	def __init__(self, processes: int, sleep_time=0.1):
		self.__processes = processes
		self.__sleep_time = sleep_time

	@abstractmethod
	def _create_worker(self) -> MonteCarloWorkerAgent:
		pass

	def __create_and_start_worker(self):
		worker = self._create_worker()
		worker.start()

	def __filter_alive_processes(self, processes):
		return [process for process in processes if process.is_alive()]

	def start(self):

		processes = []
		while True:
			processes = self.__filter_alive_processes(processes)

			while len(processes) < self.__processes:
				process = Process(target=self.__create_and_start_worker)
				process.start()
				processes.append(process)
				processes = self.__filter_alive_processes(processes)

			time.sleep(self.__sleep_time)
