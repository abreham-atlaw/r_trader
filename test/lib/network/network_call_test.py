import unittest
from requests.exceptions import HTTPError

from lib.network import network_call


class NetworkCallTest(unittest.TestCase):
	class NetworkRequests:

		def __init__(self) -> None:
			self.tries = 0

		@network_call
		def request0(self, arg0, arg2):
			return f"{arg0}-{arg2}"

		@network_call
		def request1(self, arg0, arg2):
			if self.tries < 3:
				self.tries += 1
				raise HTTPError()
			return f"{arg0}-{arg2}"

		@network_call
		def request2(self, arg0, arg2):
			raise HTTPError()

	def setUp(self) -> None:
		self.requests = NetworkCallTest.NetworkRequests()

	def test_successfull_request(self):
		response = self.requests.request0(1, 0)
		self.assertEquals(response, "1-0")

	def test_success_after_trials_request(self):
		response = self.requests.request1(1, 2)
		self.assertEqual(self.requests.tries, 3)
		self.assertEqual(response, "1-2")

	def test_failure_request(self):
		try:
			response = self.requests.request2(1, 2)
		except HTTPError:
			response = 0

		self.assertEquals(response, 0)


if __name__ == "__main__":
	unittest.main()
