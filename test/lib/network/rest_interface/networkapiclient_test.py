from typing import *

import unittest
import attr
import json

from lib.network.rest_interface import NetworkApiClient
from lib.network.rest_interface import Request



class NetworkApiClientTest(unittest.TestCase):


	@attr.define
	class User:

		id: int = attr.ib()
		name: str = attr.ib()
		email: str = attr.ib()
		gender: str = attr.ib()
		status: str = attr.ib()


	@attr.define
	class NewUserForm:

		name: str = attr.ib()
		email: str = attr.ib()
		gender: str = attr.ib()
		status: str = attr.ib()

	
	class GetUsersRequest(Request):

		def __init__(self):
			super().__init__("users/", method=Request.Method.GET, output_class=List[NetworkApiClientTest.User])
		
		def _filter_response(self, response: Dict) -> Dict:
			return response["data"]


	class CreateNewUserRequest(Request):

		def __init__(self, form):
			super().__init__("users/", method=Request.Method.POST, post_data=form, output_class=NetworkApiClientTest.User)

		def _filter_response(self, response: Dict) -> Dict:
			return response["data"]
	

	URL = "https://gorest.co.in/public/v1/"

	def setUp(self) -> None:
		self.client = NetworkApiClient(NetworkApiClientTest.URL)
	
	def test_execute_get_request(self):
		response = self.client.execute(NetworkApiClientTest.GetUsersRequest())
		self.assertIsNotNone(response)
		self.assertIsInstance(response, list)
		self.assertIsInstance(response[0], NetworkApiClientTest.User)
	
	def test_execute_post_request(self):
		form = NetworkApiClientTest.NewUserForm(
			"elliot alderson",
			"elliotalderson@mrrobot.com",
			"male",
			"active"
		)
		response: NetworkApiClientTest.User = self.client.execute(
			NetworkApiClientTest.CreateNewUserRequest(form),
			headers={
				"Authorization":"Bearer 4f1062eb69e145996bf45723739760ccf3fdcb5525959d86772430f0a0b0c234"
			}
		)
		self.assertIsNotNone(response)
		self.assertIsInstance(response, NetworkApiClientTest.User)
		self.assertEquals(response.name, form.name)


if __name__ == "__main__":
	unittest.main()
