import unittest

from core import Config
from core.di import ServiceProvider
from core.utils.resman.data.repositories.resource_repository import ResourceUnavailableException


class MongoResourceRepository(unittest.TestCase):

	def setUp(self):
		self.repository = ServiceProvider.provide_resman(Config.ResourceCategories.OANDA_SIM_ACCOUNTS)

	def test_create(self):

		ids = [
			'7c4379d6-5c8a-46d5-9d73-158e7aec5c95',
			'25662bee-8f8b-4727-a7e4-fb379e13537e',
			'810ea4d6-405e-4c37-a5b0-24429840cbea',
			'ce2f59a3-bcf9-4a0f-8258-f61809544a5c',
			'8c85090d-763a-4870-b63f-210cc5ba37e9',
			'c3bf9ad0-cb76-4e6d-875d-bf2cf3453a3b',
			'50224c87-ce99-4efd-ad14-c16ceff5f8ac',
			'f1b3c17e-a617-4d55-adbe-8628b7adad5c',
			'ec9179da-a5ff-42f3-9562-269651745305',
			'2d80193d-3d2d-4a02-9748-75aba6c03ed5',
			'9a0c6d7f-e73d-43c7-bdcd-4b2105c71fdf',
			'ed0dfa67-4246-4833-b1ed-cd6fc579423b',
			'0f3d5b8f-1884-4a0e-99a7-1a35e59f8fd9',
			'7ce02eda-30f1-4c3b-9848-367338e1ce06',
			'200f5119-4abd-44d1-ba29-6ecf746ac158',
			'b51ed6bd-206f-4fa6-863f-6d0c03c7b17b',
			'1dd25830-fb2a-43c1-8be7-d3a3df072342',
			'531fa74e-cad4-43f6-bc44-9778826ec07d',
			'e9ddbdfd-2cb2-41ac-b867-14ba114e4c21',
			'bdf3de4f-7c66-4f9f-9ed5-1f4c658d150f',
			'675ca31f-ba2d-458d-abea-702e46894f04',
			'e463c438-ea8e-40af-90b7-17d7ec93ff13',
			'099ae2ac-2210-4961-a283-513e4a7c3be7',
			'1e933215-1296-4e1c-a279-3d66c69a2014',
			'dd231849-9957-46f9-9bc0-c3218386c7cc',
			'20705b80-fe91-4303-b72d-3a71a35c3e21',
			'f899ee96-2529-4c32-bdf1-6bf981de0608',
			'1ec75c05-a362-4254-b4b2-19e119841f5d',
			'6d9e1dfb-7c1e-45b8-b50b-ea1913c435b8',
			'2370e78c-991c-412f-96bb-b236ea4aab4f'
		]

		for id in ids:
			self.repository.create(id)

	def test_allocate(self):

		allocated = []
		exhausted = False

		while not exhausted:
			try:
				allocated.append(self.repository.allocate().id)
			except ResourceUnavailableException:
				exhausted = True
		print(allocated)
		self.assertEqual(len(allocated), len(set(allocated)))

	def test_release(self):

		locked = self.repository.get_locked()

		print(f"Releasing {len(locked)} Resources")

		for resource in locked:
			self.repository.release(resource)

		self.assertEqual(len(self.repository.get_locked()), 0)
