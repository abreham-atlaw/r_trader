
import unittest as ut


from lib.concurrency import ConcurrencyHandler, ConcurrentProcess, Pool


class HandlerTest(ut.TestCase):

	MAX_PROCESSES = 4

	def function(self, y):
		print(f"Starting Process {y}")
		x = y
		for i in range(1, 70000000):
			x += i
		print(x)

		print(f"Finishing Process {y}")
		return x

	def test_max_process(self):
		handler = ConcurrencyHandler(
			HandlerTest.MAX_PROCESSES
		)
		processes = []
		for i in range(100):
			process = ConcurrentProcess(function=self.function, function_arguments=(i,))
			handler.enqueue(process)
			processes.append(process)

