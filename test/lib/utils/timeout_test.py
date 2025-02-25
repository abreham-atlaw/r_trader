import time
import unittest
from datetime import datetime

from lib.utils.decorators import timeout_function
from lib.utils.timeout import timeout


class TimeoutTest(unittest.TestCase):

	def test_functionality(self):

		def func(duration):
			time.sleep(duration)

		DURATION = 10

		start_time = datetime.now()

		timeout(
			func=lambda: func(60),
			duration=DURATION
		)

		duration = (datetime.now() - start_time).total_seconds()
		self.assertAlmostEqual(duration, DURATION, delta=1)

	def test_decorator(self):
		DURATION = 10

		start_time = datetime.now()

		@timeout_function(duration=DURATION)
		def func(duration):
			time.sleep(duration)

		func(60)

		duration = (datetime.now() - start_time).total_seconds()
		self.assertAlmostEqual(duration, DURATION, delta=1)
