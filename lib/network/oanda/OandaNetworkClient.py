from lib.network.rest_interface import NetworkApiClient
from lib.network.rest_interface import Request


class OandaNetworkClient(NetworkApiClient):

	def __init__(self, url, token, account_id, *args, **kwargs):
		super().__init__(url, *args, **kwargs)
		self.__account_id = account_id
		self.__token = token
		self.__headers = {
			"Authorization": f"Bearer {self.__token}",
		}

	def _get_complete_url(self, url):
		url = super()._get_complete_url(url)
		if "{account_id}" in url:
			url = url.format(account_id=self.__account_id)
		return url
	
	def __construct_headers(self, headers):
		if headers is None:
			headers = {}
		headers.update(self.__headers)
		return headers

	def execute(self, request: Request, headers=None):
		return super().execute(request, headers=self.__construct_headers(headers))
