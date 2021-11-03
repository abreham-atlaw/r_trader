import unittest
import attr

from lib.network.rest_interface import Request


@attr.define
class Model1:

	attrib0: str = attr.ib()
	attrib1: int = attr.ib()


@attr.define
class Model2:

	attrib0: int = attr.ib()


class RequestTest(unittest.TestCase):

	MODEL1_INSTANCE = Model1("hello", 0)
	MODEL1_JSON = '{"attrib0": "hello", "attrib1": 0}'

	MODEL2_INSTANCE = Model2(5)
	MODEL2_JSON = '{"attrib0": 5}'


	def test_url_formatting(self):
		request = Request("test/{test1}/{test2}/test3", url_params={"test1": "value1", "test2": "value2"})
		self.assertEqual(request.get_url(), "test/value1/value2/test3")
	
	def test_deserialize_object(self):
		request = Request("test/", output_class=Model1)
		self.assertEqual(request.deserialize_object(RequestTest.MODEL1_JSON), RequestTest.MODEL1_INSTANCE)
	
	def test_get_post_data(self):
		request = Request("test/", post_data=RequestTest.MODEL2_INSTANCE)
		self.assertEqual(
			request.get_post_data(),
			RequestTest.MODEL2_JSON
		)


if __name__ == "__main__":
	unittest.main()
