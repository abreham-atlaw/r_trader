import typing
from bson.objectid import ObjectId
from pymongo import MongoClient

from core.utils.resman.data.models import Resource
from .resource_repository import ResourceRepository


class MongoResourceRepository(ResourceRepository):
	
	def __init__(
			self,
			category: str,
			mongo_client: MongoClient,
			database_name: str = 'resources',
	):
		super().__init__(category)
		self.__mongo_client = mongo_client
		self.__db = self.__mongo_client[database_name]
		self.__collection = self.__db[category]

	def _get_all(self) -> typing.List[Resource]:
		resources = []
		for doc in self.__collection.find():
			resource = Resource(
				id=str(doc['_id']),
				lock_datetime=doc['lock_datetime']
			)
			resources.append(resource)
		return resources

	def _update(self, resource: Resource):
		self.__collection.update_one(
			{'_id': ObjectId(resource.id)},
			{'$set': {
				'lock_datetime': resource.lock_datetime
			}}
		)

	def _create(self, resource: Resource):
		doc = {
			'lock_datetime': resource.lock_datetime
		}
		result = self.__collection.insert_one(doc)
		resource.id = str(result.inserted_id)
